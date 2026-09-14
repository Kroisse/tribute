//! Type constraint solver.
//!
//! Solves type constraints using union-find based unification.

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
    /// Several exact-identity candidates remain possible until more types solve.
    AmbiguousEffect {
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

/// A solver error together with the source construct that introduced the
/// failing constraint, when one was recorded by the checker.
#[derive(Clone, Debug)]
pub struct LocatedSolveError<'db> {
    pub error: SolveError<'db>,
    pub origin: Option<ConstraintOrigin>,
}

/// Format an effect row using stable, source-oriented syntax.
///
/// Internal row-variable IDs are intentionally hidden.  Open rows use the
/// generic source-level tail name `e`, and concrete effects are sorted so the
/// message does not depend on insertion or hash-map iteration order.
pub fn format_effect_row(db: &dyn salsa::Database, row: EffectRow<'_>) -> String {
    let mut items: Vec<String> = row.effects(db).iter().map(ToString::to_string).collect();
    items.sort();
    if row.rest(db).is_some() {
        items.push("e".to_string());
    }
    format!("{{{}}}", items.join(", "))
}

impl std::fmt::Display for SolveError<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeMismatch { expected, actual } => {
                write!(f, "expected `{}`, found `{}`", expected, actual)
            }
            Self::OccursCheck { ty, .. } => {
                write!(
                    f,
                    "infinite type: cannot construct the infinite type `{}`",
                    ty
                )
            }
            Self::RowMismatch { expected, actual } | Self::AmbiguousEffect { expected, actual } => {
                salsa::with_attached_database(|db| {
                    write!(
                        f,
                        "effect mismatch: expected `{}`, found `{}`",
                        format_effect_row(db, *expected),
                        format_effect_row(db, *actual)
                    )
                })
                .unwrap_or(Err(std::fmt::Error))
            }
            Self::EffectArgArityMismatch {
                effect_name,
                expected,
                found,
            } => {
                use tribute_core::fmt::PluralExt;
                effect_name.with_str(|name| {
                    write!(
                        f,
                        "ability `{}` expects {} type argument{}, but {} {} given",
                        name,
                        expected,
                        expected.plural(),
                        found,
                        found.verb("was", "were"),
                    )
                })
            }
        }
    }
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
            TypeKind::Named { id, name, args } => {
                let args = args
                    .iter()
                    .map(|a| self.apply_with_rows(db, *a, row_subst))
                    .collect();
                Type::new(
                    db,
                    TypeKind::Named {
                        id: *id,
                        name: *name,
                        args,
                    },
                )
            }
            TypeKind::Func {
                params,
                result,
                effect,
                minimum_convention,
            } => {
                let params = params
                    .iter()
                    .map(|p| self.apply_with_rows(db, *p, row_subst))
                    .collect();
                let result = self.apply_with_rows(db, *result, row_subst);
                let row_applied = row_subst.apply(db, *effect);
                let effect = map_effect_row_type_args(db, row_applied, |a| {
                    self.apply_with_rows(db, a, row_subst)
                });
                Type::new(
                    db,
                    TypeKind::Func {
                        params,
                        result,
                        effect,
                        minimum_convention: *minimum_convention,
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
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => {
                let arg = self.apply_with_rows(db, *arg, row_subst);
                let result = self.apply_with_rows(db, *result, row_subst);
                let row_applied = row_subst.apply(db, *effect);
                let effect = map_effect_row_type_args(db, row_applied, |a| {
                    self.apply_with_rows(db, a, row_subst)
                });
                Type::new(
                    db,
                    TypeKind::Continuation {
                        arg,
                        result,
                        effect,
                    },
                )
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

    /// Generalize unresolved variables except those free in the environment.
    pub fn generalize_excluding(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
        excluded: &[UniVarId<'db>],
    ) -> (Type<'db>, Vec<TypeParam>) {
        let (generalized, type_params, _) =
            self.generalize_excluding_with_mapping(db, ty, row_subst, excluded);
        (generalized, type_params)
    }

    /// Generalize unresolved variables except those free in the environment,
    /// retaining the variable-to-local-scheme mapping for typed-body metadata.
    pub fn generalize_excluding_with_mapping(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
        excluded: &[UniVarId<'db>],
    ) -> (Type<'db>, Vec<TypeParam>, HashMap<UniVarId<'db>, u32>) {
        let mut univars = Vec::new();
        self.collect_unresolved_univars(db, ty, row_subst, &mut univars);
        univars.retain(|id| !excluded.contains(id));

        let var_to_index: HashMap<UniVarId<'db>, u32> = univars
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index as u32))
            .collect();
        let generalized = self.replace_univars_with_bound(db, ty, row_subst, &var_to_index);
        let type_params = univars.iter().map(|_| TypeParam::anonymous()).collect();
        (generalized, type_params, var_to_index)
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

    /// Finalize a typed body or its semantic metadata.
    ///
    /// Function-interface variables become `BoundVar`; variables owned by a
    /// local generalized scheme become `LocalBoundVar`.
    pub(crate) fn apply_generalization_with_local_vars(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
        local_vars: &HashMap<UniVarId<'db>, (crate::ast::NodeId, u32)>,
    ) -> Type<'db> {
        self.replace_univars_with_bound_and_local(db, ty, row_subst, var_to_index, local_vars)
    }

    fn replace_univars_with_bound_and_local(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
        local_vars: &HashMap<UniVarId<'db>, (crate::ast::NodeId, u32)>,
    ) -> Type<'db> {
        match ty.kind(db) {
            TypeKind::UniVar { id } => {
                if let Some(&(scope, index)) = local_vars.get(id) {
                    Type::new(db, TypeKind::LocalBoundVar { scope, index })
                } else if let Some(&index) = var_to_index.get(id) {
                    Type::new(db, TypeKind::BoundVar { index })
                } else if let Some(subst_ty) = self.get(*id) {
                    self.replace_univars_with_bound_and_local(
                        db,
                        subst_ty,
                        row_subst,
                        var_to_index,
                        local_vars,
                    )
                } else {
                    ty
                }
            }
            TypeKind::Named { id, name, args } => Type::new(
                db,
                TypeKind::Named {
                    id: *id,
                    name: *name,
                    args: args
                        .iter()
                        .map(|arg| {
                            self.replace_univars_with_bound_and_local(
                                db,
                                *arg,
                                row_subst,
                                var_to_index,
                                local_vars,
                            )
                        })
                        .collect(),
                },
            ),
            TypeKind::Func {
                params,
                result,
                effect,
                minimum_convention,
            } => Type::new(
                db,
                TypeKind::Func {
                    params: params
                        .iter()
                        .map(|param| {
                            self.replace_univars_with_bound_and_local(
                                db,
                                *param,
                                row_subst,
                                var_to_index,
                                local_vars,
                            )
                        })
                        .collect(),
                    result: self.replace_univars_with_bound_and_local(
                        db,
                        *result,
                        row_subst,
                        var_to_index,
                        local_vars,
                    ),
                    effect: map_effect_row_type_args(db, row_subst.apply(db, *effect), |arg| {
                        self.replace_univars_with_bound_and_local(
                            db,
                            arg,
                            row_subst,
                            var_to_index,
                            local_vars,
                        )
                    }),
                    minimum_convention: *minimum_convention,
                },
            ),
            TypeKind::Tuple(elements) => Type::new(
                db,
                TypeKind::Tuple(
                    elements
                        .iter()
                        .map(|element| {
                            self.replace_univars_with_bound_and_local(
                                db,
                                *element,
                                row_subst,
                                var_to_index,
                                local_vars,
                            )
                        })
                        .collect(),
                ),
            ),
            TypeKind::App { ctor, args } => Type::new(
                db,
                TypeKind::App {
                    ctor: self.replace_univars_with_bound_and_local(
                        db,
                        *ctor,
                        row_subst,
                        var_to_index,
                        local_vars,
                    ),
                    args: args
                        .iter()
                        .map(|arg| {
                            self.replace_univars_with_bound_and_local(
                                db,
                                *arg,
                                row_subst,
                                var_to_index,
                                local_vars,
                            )
                        })
                        .collect(),
                },
            ),
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => Type::new(
                db,
                TypeKind::Continuation {
                    arg: self.replace_univars_with_bound_and_local(
                        db,
                        *arg,
                        row_subst,
                        var_to_index,
                        local_vars,
                    ),
                    result: self.replace_univars_with_bound_and_local(
                        db,
                        *result,
                        row_subst,
                        var_to_index,
                        local_vars,
                    ),
                    effect: map_effect_row_type_args(db, row_subst.apply(db, *effect), |entry| {
                        self.replace_univars_with_bound_and_local(
                            db,
                            entry,
                            row_subst,
                            var_to_index,
                            local_vars,
                        )
                    }),
                },
            ),
            TypeKind::BoundVar { .. }
            | TypeKind::LocalBoundVar { .. }
            | TypeKind::Int
            | TypeKind::Nat
            | TypeKind::Float
            | TypeKind::Bool
            | TypeKind::Bytes
            | TypeKind::Rune
            | TypeKind::Nil
            | TypeKind::Never
            | TypeKind::Error => ty,
        }
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
                ..
            } => {
                for p in params {
                    self.collect_unresolved_univars(db, *p, row_subst, out);
                }
                self.collect_unresolved_univars(db, *result, row_subst, out);
                self.collect_univars_from_effect_row(db, *effect, row_subst, out);
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
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => {
                self.collect_unresolved_univars(db, *arg, row_subst, out);
                self.collect_unresolved_univars(db, *result, row_subst, out);
                self.collect_univars_from_effect_row(db, *effect, row_subst, out);
            }
            _ => {}
        }
    }

    /// Collect unresolved UniVarIds from an effect row's type arguments.
    fn collect_univars_from_effect_row(
        &self,
        db: &'db dyn salsa::Database,
        effect: EffectRow<'db>,
        row_subst: &RowSubst<'db>,
        out: &mut Vec<UniVarId<'db>>,
    ) {
        let applied = row_subst.apply(db, effect);
        for e in applied.effects(db) {
            for a in &e.args {
                self.collect_unresolved_univars(db, *a, row_subst, out);
            }
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
        self.replace_univars_with_bound_and_local(db, ty, row_subst, var_to_index, &HashMap::new())
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
        let mut row = row;
        let mut visited = SmallVec::<[u64; 8]>::new();
        while let Some(var) = row.rest(db) {
            let Some(subst_row) = self.get(var.id) else {
                break;
            };
            if visited.contains(&var.id) {
                break; // cycle detected — return current row as-is
            }
            visited.push(var.id);

            let mut effects = row.effects(db).clone();
            for effect in subst_row.effects(db) {
                if !effects.contains(effect) {
                    effects.push(effect.clone());
                }
            }
            let rest = subst_row.rest(db);
            row = EffectRow::new(db, effects, rest);
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

    fn union_variables(
        &self,
        union: &crate::ast::RowUnion<'db>,
    ) -> (Vec<UniVarId<'db>>, Vec<EffectVar>) {
        let mut types = Vec::new();
        let mut rows = Vec::new();
        for row in union.sources.iter().chain(std::iter::once(&union.result)) {
            let row = self.normalize_row(*row);
            if let Some(var) = row.rest(self.db)
                && !rows.contains(&var)
            {
                rows.push(var);
            }
            for effect in row.effects(self.db) {
                for arg in &effect.args {
                    self.type_subst.collect_univars_from_type(
                        self.db,
                        *arg,
                        &self.row_subst,
                        &mut types,
                    );
                    for var in collect_effect_vars(self.db, *arg) {
                        if !rows.contains(&var) {
                            rows.push(var);
                        }
                    }
                }
            }
        }
        (types, rows)
    }

    pub fn expand_union_dependencies(
        &self,
        types: &mut Vec<UniVarId<'db>>,
        rows: &mut Vec<EffectVar>,
    ) {
        loop {
            let before = (types.len(), rows.len());
            for union in self.row_dependency_views() {
                let (ut, ur) = self.union_variables(&union);
                if ut.iter().any(|v| types.contains(v)) || ur.iter().any(|v| rows.contains(v)) {
                    for var in ut {
                        if !types.contains(&var) {
                            types.push(var);
                        }
                    }
                    for var in ur {
                        if !rows.contains(&var) {
                            rows.push(var);
                        }
                    }
                }
            }
            if before == (types.len(), rows.len()) {
                break;
            }
        }
    }

    /// Project away unconstrained internal row components. A component made
    /// solely of open, label-free unions with at most one externally visible
    /// variable imposes no restriction: assigning that row to every variable
    /// satisfies every equation. Keeping it would export implementation-local
    /// existential variables and multiply them at every call.
    fn row_relations_for_type(
        &self,
        ty: Type<'db>,
    ) -> (
        Vec<crate::ast::RowUnion<'db>>,
        Vec<crate::ast::RowRemoval<'db>>,
    ) {
        let ty = self
            .type_subst
            .apply_with_rows(self.db, ty, &self.row_subst);
        let visible = collect_effect_vars(self.db, ty);
        let mut visible_types = Vec::new();
        self.type_subst
            .collect_univars_from_type(self.db, ty, &self.row_subst, &mut visible_types);
        let union_count = self.pending_row_unions.len();
        let removals = self.retained_row_removals();
        let unions = self.row_dependency_views();
        let variables: Vec<_> = unions
            .iter()
            .map(|union| self.union_variables(union))
            .collect();
        let mut visited = vec![false; unions.len()];
        let mut retained = Vec::new();
        for start in 0..unions.len() {
            if visited[start] {
                continue;
            }
            let mut component = vec![start];
            let (mut types, mut vars) = variables[start].clone();
            visited[start] = true;
            loop {
                let before = component.len();
                for (i, (union_types, rows)) in variables.iter().enumerate() {
                    if visited[i] {
                        continue;
                    }
                    if rows.iter().any(|v| vars.contains(v))
                        || union_types.iter().any(|v| types.contains(v))
                    {
                        visited[i] = true;
                        component.push(i);
                        for var in rows {
                            if !vars.contains(var) {
                                vars.push(*var);
                            }
                        }
                        for var in union_types {
                            if !types.contains(var) {
                                types.push(*var);
                            }
                        }
                    }
                }
                if component.len() == before {
                    break;
                }
            }
            // A constraint disconnected from the generalized interface belongs
            // to this solver, not to every later let binding in the function.
            if !vars.iter().any(|var| visible.contains(var))
                && !types.iter().any(|var| visible_types.contains(var))
            {
                continue;
            }
            let unrestricted = component.iter().all(|i| {
                let union = &unions[*i];
                *i < union_count
                    && union.result.rest(self.db).is_some()
                    && union.sources.iter().any(|row| row.rest(self.db).is_some())
                    && union
                        .sources
                        .iter()
                        .chain(std::iter::once(&union.result))
                        .all(|row| row.effects(self.db).is_empty())
            });
            if unrestricted && vars.iter().filter(|var| visible.contains(var)).count() <= 1 {
                continue;
            }
            retained.extend(component);
        }
        (
            retained
                .iter()
                .filter(|i| **i < union_count)
                .map(|i| unions[*i].clone())
                .collect(),
            retained
                .iter()
                .filter(|i| **i >= union_count)
                .map(|i| removals[*i - union_count].clone())
                .collect(),
        )
    }

    pub fn row_unions_for_type(&self, ty: Type<'db>) -> Vec<crate::ast::RowUnion<'db>> {
        self.row_relations_for_type(ty).0
    }
    pub fn row_removals_for_type(&self, ty: Type<'db>) -> Vec<crate::ast::RowRemoval<'db>> {
        self.row_relations_for_type(ty).1
    }

    // Uniform row lists for dependency analysis only. Subtractions are never
    // sent to the union solver through these graph views.
    fn row_dependency_views(&self) -> Vec<crate::ast::RowUnion<'db>> {
        self.retained_row_unions()
            .into_iter()
            .chain(
                self.retained_row_removals()
                    .into_iter()
                    .map(|r| crate::ast::RowUnion {
                        sources: vec![r.source, r.removed],
                        result: r.result,
                    }),
            )
            .collect()
    }

    pub fn row_removal_variables(
        &self,
        removals: &[crate::ast::RowRemoval<'db>],
    ) -> (Vec<UniVarId<'db>>, Vec<EffectVar>) {
        self.row_union_variables(
            &removals
                .iter()
                .map(|r| crate::ast::RowUnion {
                    sources: vec![r.source, r.removed],
                    result: r.result,
                })
                .collect::<Vec<_>>(),
        )
    }

    pub fn row_union_variables(
        &self,
        unions: &[crate::ast::RowUnion<'db>],
    ) -> (Vec<UniVarId<'db>>, Vec<EffectVar>) {
        let mut types = Vec::new();
        let mut rows = Vec::new();
        for union in unions {
            let (ut, ur) = self.union_variables(union);
            for var in ut {
                if !types.contains(&var) {
                    types.push(var);
                }
            }
            for var in ur {
                if !rows.contains(&var) {
                    rows.push(var);
                }
            }
        }
        (types, rows)
    }

    pub fn add_row_unions(&mut self, unions: Vec<crate::ast::RowUnion<'db>>) {
        for union in unions {
            for row in union.sources.iter().chain(std::iter::once(&union.result)) {
                self.reserve_effect_vars_in_row(*row);
            }
            if !self.pending_row_unions.iter().any(|(old, _)| *old == union) {
                self.pending_row_unions.push((union, None));
            }
        }
    }

    pub fn generalize_row_union(
        &self,
        union: &crate::ast::RowUnion<'db>,
        mapping: &HashMap<UniVarId<'db>, u32>,
    ) -> crate::ast::RowUnion<'db> {
        let mut union = union.clone();
        union.for_each_row_mut(|row| {
            *row = map_effect_row_type_args(self.db, self.normalize_row(*row), |ty| {
                self.type_subst
                    .apply_generalization(self.db, ty, &self.row_subst, mapping)
            });
        });
        union
    }

    /// Relations which must be quantified together with the enclosing scheme.
    pub fn retained_row_unions(&self) -> Vec<crate::ast::RowUnion<'db>> {
        self.pending_row_unions
            .iter()
            .map(|(union, _)| {
                let mut union = union.clone();
                union.for_each_row_mut(|row| *row = self.normalize_row(*row));
                union
            })
            .collect()
    }

    fn normalize_row(&self, row: EffectRow<'db>) -> EffectRow<'db> {
        let row = self.row_subst.apply(self.db, row);
        map_effect_row_type_args(self.db, row, |ty| {
            self.type_subst
                .apply_with_rows(self.db, ty, &self.row_subst)
        })
    }

    fn settle_row_unions(&mut self) -> Result<(), LocatedSolveError<'db>> {
        let mut first_error = None;
        for (union, origin) in std::mem::take(&mut self.pending_row_unions) {
            match self.solve_row_union(&union) {
                Ok(true) => {}
                Ok(false) => self.pending_row_unions.push((union, origin)),
                Err(error) => {
                    first_error.get_or_insert(LocatedSolveError { error, origin });
                }
            }
        }
        for (removal, origin) in std::mem::take(&mut self.pending_row_removals) {
            match self.solve_row_removal(&removal) {
                Ok(true) => {}
                Ok(false) => self.pending_row_removals.push((removal, origin)),
                Err(error) => {
                    first_error.get_or_insert(LocatedSolveError { error, origin });
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub fn add_row_removals(&mut self, removals: Vec<crate::ast::RowRemoval<'db>>) {
        for removal in removals {
            for row in removal.rows() {
                self.reserve_effect_vars_in_row(row);
            }
            if !self
                .pending_row_removals
                .iter()
                .any(|(old, _)| *old == removal)
            {
                self.pending_row_removals.push((removal, None));
            }
        }
    }

    pub fn retained_row_removals(&self) -> Vec<crate::ast::RowRemoval<'db>> {
        self.pending_row_removals
            .iter()
            .map(|(removal, _)| {
                let mut removal = removal.clone();
                removal.for_each_row_mut(|row| *row = self.normalize_row(*row));
                removal
            })
            .collect()
    }

    pub fn generalize_row_removal(
        &self,
        removal: &crate::ast::RowRemoval<'db>,
        mapping: &HashMap<UniVarId<'db>, u32>,
    ) -> crate::ast::RowRemoval<'db> {
        let mut removal = removal.clone();
        removal.for_each_row_mut(|row| {
            *row = map_effect_row_type_args(self.db, self.normalize_row(*row), |ty| {
                self.type_subst
                    .apply_generalization(self.db, ty, &self.row_subst, mapping)
            });
        });
        removal
    }

    fn solve_row_removal(
        &mut self,
        removal: &crate::ast::RowRemoval<'db>,
    ) -> Result<bool, SolveError<'db>> {
        let mut removal = removal.clone();
        removal.for_each_row_mut(|row| *row = self.normalize_row(*row));
        assert!(
            removal.removed.rest(self.db).is_none(),
            "handler removal set must be closed"
        );
        if removal
            .result
            .effects(self.db)
            .iter()
            .any(|effect| removal.removed.effects(self.db).contains(effect))
        {
            let allowed: Vec<_> = removal
                .result
                .effects(self.db)
                .iter()
                .filter(|effect| !removal.removed.effects(self.db).contains(effect))
                .cloned()
                .collect();
            return Err(SolveError::RowMismatch {
                expected: EffectRow::new(self.db, allowed, removal.result.rest(self.db)),
                actual: removal.result,
            });
        }
        let mut remaining = Vec::new();
        let mut ambiguous = false;
        for effect in removal.source.effects(self.db) {
            if removal.removed.effects(self.db).contains(effect) {
                continue;
            }
            if removal.removed.effects(self.db).iter().any(|candidate| {
                candidate.ability_id == effect.ability_id
                    && candidate.args.len() == effect.args.len()
                    && candidate
                        .args
                        .iter()
                        .zip(&effect.args)
                        .all(|(a, b)| self.types_unifiable(*a, *b))
            }) {
                ambiguous = true;
            } else {
                remaining.push(effect.clone());
            }
        }
        let known = EffectRow::new(self.db, remaining, None);
        if removal.source.rest(self.db).is_none() && !ambiguous {
            self.unify_rows(removal.result, known)?;
            return Ok(true);
        }
        // Propagate only effects proven not to be removed. Do not decide the
        // membership of an unresolved type argument by unification.
        if let Some(tail) = removal.result.rest(self.db) {
            let missing: Vec<_> = known
                .effects(self.db)
                .iter()
                .filter(|effect| !removal.result.effects(self.db).contains(effect))
                .cloned()
                .collect();
            if !missing.is_empty() {
                let fresh = self.fresh_row_var();
                self.row_subst
                    .insert(tail.id, EffectRow::new(self.db, missing, Some(fresh)));
            }
        } else {
            for effect in known.effects(self.db) {
                let candidates: Vec<_> = removal
                    .result
                    .effects(self.db)
                    .iter()
                    .filter(|other| {
                        other.ability_id == effect.ability_id
                            && other.args.len() == effect.args.len()
                            && other
                                .args
                                .iter()
                                .zip(&effect.args)
                                .all(|(a, b)| self.types_unifiable(*a, *b))
                    })
                    .cloned()
                    .collect();
                match candidates.as_slice() {
                    [] => {
                        return Err(SolveError::RowMismatch {
                            expected: removal.result,
                            actual: known,
                        });
                    }
                    [other] => {
                        for (a, b) in other.args.iter().zip(&effect.args) {
                            self.unify_types(*a, *b)?;
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(false)
    }

    fn solve_row_union(
        &mut self,
        union: &crate::ast::RowUnion<'db>,
    ) -> Result<bool, SolveError<'db>> {
        let mut union = union.clone();
        union.for_each_row_mut(|row| *row = self.normalize_row(*row));
        let mut known = Vec::new();
        let mut tails = Vec::new();
        for source in &union.sources {
            for effect in source.effects(self.db) {
                if !known.contains(effect) {
                    known.push(effect.clone());
                }
            }
            if let Some(tail) = source.rest(self.db)
                && !tails.contains(&tail)
            {
                tails.push(tail);
            }
        }
        if tails.len() <= 1 {
            let actual = EffectRow::new(self.db, known, tails.first().copied());
            if actual == union.result {
                return Ok(true);
            }
            self.unify_rows(union.result, actual)?;
            return Ok(true);
        }
        if union.result.is_pure(self.db) {
            for source in union.sources {
                self.unify_rows(union.result, source)?;
            }
            return Ok(true);
        }
        if union.result.rest(self.db).is_none() {
            // A closed result is an upper bound for every source. Only commit
            // a type substitution when its instance match is unambiguous.
            for effect in &known {
                let candidates: Vec<_> = union
                    .result
                    .effects(self.db)
                    .iter()
                    .filter(|candidate| {
                        candidate.ability_id == effect.ability_id
                            && candidate.args.len() == effect.args.len()
                            && candidate
                                .args
                                .iter()
                                .zip(&effect.args)
                                .all(|(a, b)| self.types_unifiable(*a, *b))
                    })
                    .cloned()
                    .collect();
                match candidates.as_slice() {
                    [] => {
                        return Err(SolveError::RowMismatch {
                            expected: union.result,
                            actual: EffectRow::new(self.db, known.clone(), None),
                        });
                    }
                    [candidate] => {
                        for (a, b) in candidate.args.iter().zip(&effect.args) {
                            self.unify_types(*a, *b)?;
                        }
                    }
                    _ => {}
                }
            }
        } else {
            let missing: Vec<_> = known
                .into_iter()
                .filter(|effect| !union.result.effects(self.db).contains(effect))
                .collect();
            if !missing.is_empty() {
                let rest = union.result.rest(self.db).expect("open result");
                let fresh = self.fresh_row_var();
                self.row_subst
                    .insert(rest.id, EffectRow::new(self.db, missing, Some(fresh)));
            }
        }
        // There may be several valid ways to distribute the remaining labels.
        // Preserve the union instead of selecting an arbitrary source tail.
        Ok(false)
    }

    /// Nested cases in a resumptive arm can refer back to the handle answer.
    /// Resolve a closed strongly connected group from its independent sources.
    fn resolve_join_cycle(&mut self) -> Result<bool, LocatedSolveError<'db>> {
        let joins: Vec<_> = self
            .pending_relations
            .iter()
            .enumerate()
            .filter_map(|(index, relation)| {
                let Constraint::TypeJoin {
                    result,
                    sources,
                    complete: true,
                    origin,
                } = relation
                else {
                    return None;
                };
                let result = self.type_subst.apply(self.db, *result);
                matches!(result.kind(self.db), TypeKind::UniVar { .. }).then(|| {
                    (
                        index,
                        result,
                        sources
                            .iter()
                            .map(|(source, origin)| {
                                (self.type_subst.apply(self.db, *source), *origin)
                            })
                            .collect::<Vec<_>>(),
                        *origin,
                    )
                })
            })
            .collect();
        let reachable = |start: usize| {
            let mut seen = vec![false; joins.len()];
            let mut work = vec![start];
            while let Some(index) = work.pop() {
                if std::mem::replace(&mut seen[index], true) {
                    continue;
                }
                for (source, _) in &joins[index].2 {
                    for (next, (_, result, _, _)) in joins.iter().enumerate() {
                        if source == result && !seen[next] {
                            work.push(next);
                        }
                    }
                }
            }
            seen
        };
        let reachability: Vec<_> = (0..joins.len()).map(reachable).collect();
        let protected: Vec<_> = self
            .pending_producers
            .iter()
            .map(|producer| self.type_subst.apply(self.db, producer.result))
            .chain(
                self.pending_relations
                    .iter()
                    .filter_map(|relation| match relation {
                        Constraint::TypeJoin { result, .. } => {
                            Some(self.type_subst.apply(self.db, *result))
                        }
                        _ => None,
                    }),
            )
            .collect();
        for (start, reachable_from_start) in reachability.iter().enumerate() {
            let component: Vec<_> = (0..joins.len())
                .filter(|next| reachable_from_start[*next] && reachability[*next][start])
                .collect();
            if component.len() < 2 {
                continue;
            }
            let members: Vec<_> = component.iter().map(|index| joins[*index].1).collect();
            if self
                .pending_producers
                .iter()
                .any(|producer| members.contains(&self.type_subst.apply(self.db, producer.result)))
            {
                continue;
            }
            let sources: Vec<_> = component
                .iter()
                .flat_map(|index| joins[*index].2.iter().copied())
                .filter(|(source, _)| {
                    !members.contains(source) && !matches!(source.kind(self.db), TypeKind::Never)
                })
                .collect();
            if sources.iter().any(|(source, _)| protected.contains(source)) {
                continue;
            }
            // Consume the component once, including on failure, so a later
            // producer round cannot emit the same diagnostic again.
            let indices: Vec<_> = component.iter().map(|index| joins[*index].0).collect();
            let mut index = 0;
            self.pending_relations.retain(|_| {
                let keep = !indices.contains(&index);
                index += 1;
                keep
            });
            let common = sources.first().map_or_else(
                || Type::new(self.db, TypeKind::Never),
                |(source, _)| *source,
            );
            for (source, origin) in sources {
                self.unify_types(common, source)
                    .map_err(|error| LocatedSolveError {
                        error,
                        origin: Some(origin),
                    })?;
            }
            for member in component {
                self.unify_types(joins[member].1, common)
                    .map_err(|error| LocatedSolveError {
                        error,
                        origin: Some(joins[member].3),
                    })?;
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn solve_join(
        &mut self,
        sources: &[(Type<'db>, ConstraintOrigin)],
        result: Type<'db>,
        origin: ConstraintOrigin,
        complete: bool,
        finalize: bool,
        protected: &[Type<'db>],
    ) -> Result<bool, LocatedSolveError<'db>> {
        if !complete {
            return Ok(false);
        }
        let result = self.type_subst.apply(self.db, result);
        if matches!(result.kind(self.db), TypeKind::UniVar { .. })
            && self
                .pending_producers
                .iter()
                .any(|producer| self.type_subst.apply(self.db, producer.result) == result)
        {
            return Ok(false);
        }
        let mut common = None;
        let mut unknown = Vec::new();
        for (source, source_origin) in sources {
            let source = self.type_subst.apply(self.db, *source);
            // A resumption returns this very answer; it supplies no independent
            // evidence of an inhabited result (including through nested joins).
            if source == result || matches!(source.kind(self.db), TypeKind::Never) {
                continue;
            }
            if matches!(source.kind(self.db), TypeKind::UniVar { .. }) {
                unknown.push((source, *source_origin));
            } else if let Some(common) = common {
                self.unify_types(common, source)
                    .map_err(|error| LocatedSolveError {
                        error,
                        origin: Some(*source_origin),
                    })?;
            } else {
                common = Some(source);
            }
        }
        if let Some(common) = common {
            self.unify_types(result, common)
                .map_err(|error| LocatedSolveError {
                    error,
                    origin: Some(origin),
                })?;
        }
        if !unknown.is_empty() {
            if !finalize || unknown.iter().any(|(ty, _)| protected.contains(ty)) {
                return Ok(false);
            }
            let common = common.unwrap_or(unknown[0].0);
            for (source, origin) in unknown {
                self.unify_types(common, source)
                    .map_err(|error| LocatedSolveError {
                        error,
                        origin: Some(origin),
                    })?;
            }
            self.unify_types(result, common)
                .map_err(|error| LocatedSolveError {
                    error,
                    origin: Some(origin),
                })?;
        } else if common.is_none() {
            // If a previous round selected a concrete result, a source may now
            // equal it. That source remains independent evidence; only an
            // unresolved answer variable is a circular resumption reference.
            let has_result_source = sources
                .iter()
                .any(|(source, _)| self.type_subst.apply(self.db, *source) == result);
            if !has_result_source || matches!(result.kind(self.db), TypeKind::UniVar { .. }) {
                self.unify_types(result, Type::new(self.db, TypeKind::Never))
                    .map_err(|error| LocatedSolveError {
                        error,
                        origin: Some(origin),
                    })?;
            }
        }
        Ok(true)
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

    /// Apply type substitution to effect arguments in a row.
    ///
    /// This ensures that UniVars in effect args (like `State(s)` where `s` is a UniVar)
    /// are resolved before row comparison. Without this, two rows with the same ability
    /// but different (unresolved vs resolved) args would fail to unify.
    fn apply_type_subst_to_row(&self, row: EffectRow<'db>) -> EffectRow<'db> {
        map_effect_row_type_args(self.db, row, |a| self.type_subst.apply(self.db, a))
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
    fn occurs_in_effect_row(&self, var: UniVarId<'db>, effect: EffectRow<'db>) -> bool {
        self.row_subst
            .apply(self.db, effect)
            .effects(self.db)
            .iter()
            .any(|effect| effect.args.iter().any(|arg| self.occurs_in(var, *arg)))
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
                ..
            } => {
                self.row_occurs_in(var, *effect)
                    || params.iter().any(|p| self.row_occurs_in_type(var, *p))
                    || self.row_occurs_in_type(var, *result)
            }
            TypeKind::Named { args, .. } => args.iter().any(|a| self.row_occurs_in_type(var, *a)),
            TypeKind::Tuple(elems) => elems.iter().any(|e| self.row_occurs_in_type(var, *e)),
            TypeKind::App { ctor, args } => {
                self.row_occurs_in_type(var, *ctor)
                    || args.iter().any(|a| self.row_occurs_in_type(var, *a))
            }
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => {
                self.row_occurs_in_type(var, *arg)
                    || self.row_occurs_in_type(var, *result)
                    || self.row_occurs_in(var, *effect)
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
        left: EffectRow<'db>,
        right: EffectRow<'db>,
    ) -> Result<(), SolveError<'db>> {
        match self.unify_rows_inner(left, right) {
            Err(SolveError::AmbiguousEffect { .. }) => {
                let union = crate::ast::RowUnion {
                    sources: vec![left],
                    result: right,
                };
                if !self.pending_row_unions.iter().any(|(old, _)| *old == union) {
                    self.pending_row_unions.push((union, None));
                }
                Ok(())
            }
            result => result,
        }
    }

    fn unify_rows_inner(
        &mut self,
        r1: EffectRow<'db>,
        r2: EffectRow<'db>,
    ) -> Result<(), SolveError<'db>> {
        // Apply current row substitution first
        let r1 = self.row_subst.apply(self.db, r1);
        let r2 = self.row_subst.apply(self.db, r2);

        // Apply type substitution to effect args
        // This is important because effect args may contain UniVars that have been
        // resolved by previous type constraints. For example, State(s) where s was
        // already unified with Int should become State(Int) before row comparison.
        let r1 = self.apply_type_subst_to_row(r1);
        let r2 = self.apply_type_subst_to_row(r2);

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
                let missing = self.compute_effect_difference_with_unify(effects2, effects1)?;
                if !missing.is_empty() {
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
                let missing = self.compute_effect_difference_with_unify(effects1, effects2)?;
                if !missing.is_empty() {
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
        // Equality is bidirectional set membership, including idempotence
        // after type substitution. Never choose the first of several possible
        // instances of the same ability.
        let mut pairs = Vec::new();
        for (source, target) in [(effects1, effects2), (effects2, effects1)] {
            for effect in source {
                let normalized = |candidate: &crate::ast::Effect<'db>| crate::ast::Effect {
                    ability_id: candidate.ability_id,
                    args: candidate
                        .args
                        .iter()
                        .map(|ty| {
                            self.type_subst
                                .apply_with_rows(self.db, *ty, &self.row_subst)
                        })
                        .collect(),
                };
                let effect = normalized(effect);
                let mut candidates = Vec::new();
                for other in target {
                    let other = normalized(other);
                    if other.ability_id != effect.ability_id {
                        continue;
                    }
                    if other.args.len() != effect.args.len() {
                        return Err(SolveError::EffectArgArityMismatch {
                            effect_name: effect.ability_id.name(self.db),
                            expected: effect.args.len(),
                            found: other.args.len(),
                        });
                    }
                    if effect == other {
                        candidates = vec![other];
                        break;
                    }
                    if other
                        .args
                        .iter()
                        .zip(&effect.args)
                        .all(|(a, b)| self.types_unifiable(*a, *b))
                        && !candidates.contains(&other)
                    {
                        candidates.push(other);
                    }
                }
                match candidates.as_slice() {
                    [other] => {
                        for (a, b) in effect.args.iter().zip(&other.args) {
                            pairs.push((*a, *b));
                        }
                    }
                    [] => {
                        return Err(SolveError::RowMismatch {
                            expected: r1,
                            actual: r2,
                        });
                    }
                    _ => {
                        return Err(SolveError::AmbiguousEffect {
                            expected: r1,
                            actual: r2,
                        });
                    }
                }
            }
        }

        // Do not let early matches erase ambiguity in the reverse direction.
        for (a, b) in pairs {
            self.unify_types(a, b)?;
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
            let e2 = self
                .normalize_row(EffectRow::single(self.db, e2.clone()))
                .effects(self.db)[0]
                .clone();
            let mut candidates = Vec::new();
            for e1 in list1 {
                let e1 = self
                    .normalize_row(EffectRow::single(self.db, e1.clone()))
                    .effects(self.db)[0]
                    .clone();
                if e1.ability_id != e2.ability_id {
                    continue;
                }
                if e1.args.len() != e2.args.len() {
                    return Err(SolveError::EffectArgArityMismatch {
                        effect_name: e1.ability_id.name(self.db),
                        expected: e1.args.len(),
                        found: e2.args.len(),
                    });
                }
                if e1 == e2 {
                    candidates = vec![e1];
                    break;
                }
                if e1
                    .args
                    .iter()
                    .zip(&e2.args)
                    .all(|(a, b)| self.types_unifiable(*a, *b))
                    && !candidates.contains(&e1)
                {
                    candidates.push(e1);
                }
            }
            match candidates.as_slice() {
                [e1] => {
                    for (a, b) in e1.args.iter().zip(&e2.args) {
                        self.unify_types(*a, *b)?;
                    }
                }
                [] => {
                    if !only_list2.contains(&e2) {
                        only_list2.push(e2);
                    }
                }
                _ => {
                    return Err(SolveError::AmbiguousEffect {
                        expected: EffectRow::new(self.db, list1.to_vec(), None),
                        actual: EffectRow::new(self.db, list2.to_vec(), None),
                    });
                }
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
        let only_list1 = self.compute_effect_difference_with_unify(list2, list1)?;
        let only_list2 = self.compute_effect_difference_with_unify(list1, list2)?;
        Ok((only_list1, only_list2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AbilityId, Effect, EffectRow, TypeDefId, UniVarSource};
    use trunk_ir::Symbol;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    /// Create an AbilityId for testing (with empty module path).
    fn test_ability_id<'db>(db: &'db dyn salsa::Database, name: &str) -> AbilityId<'db> {
        AbilityId::source(db, Symbol::from_dynamic(name))
    }

    /// Create a fresh type variable for testing.
    fn fresh_var(db: &dyn salsa::Database, n: u64) -> Type<'_> {
        let source = UniVarSource::Anonymous(n);
        let id = UniVarId::new(db, source, 0);
        Type::new(db, TypeKind::UniVar { id })
    }

    #[test]
    fn fresh_row_var_avoids_rows_already_present_in_constraints() {
        let db = test_db();
        let existing = EffectVar { id: 1000 };
        let existing_row = EffectRow::open(&db, existing);
        let mut constraints = ConstraintSet::new();
        constraints.add_row_eq(existing_row, existing_row);
        let mut solver = TypeSolver::new(&db);

        solver.solve(constraints).expect("constraint should solve");

        assert_eq!(solver.fresh_row_var(), EffectVar { id: 1001 });
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
                id: TypeDefId::builtin_list(&db),
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
                ability_id: test_ability_id(&db, "State"),
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
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        let result = solver.unify_types(var_ty, func_ty);
        assert!(
            matches!(result, Err(SolveError::OccursCheck { .. })),
            "Expected occurs check failure for ?a = fn() ->{{State(?a)}} Int"
        );
    }

    #[test]
    fn test_occurs_check_applies_effect_row_substitution() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let var_ty = fresh_var(&db, 0);
        let TypeKind::UniVar { id: var_id } = var_ty.kind(&db) else {
            unreachable!("fresh_var must create a UniVar")
        };
        let row_var = EffectVar { id: 7 };
        let substituted = EffectRow::new(
            &db,
            vec![Effect {
                ability_id: test_ability_id(&db, "State"),
                args: vec![var_ty],
            }],
            None,
        );
        solver.row_subst.insert(row_var.id, substituted);

        let int_ty = Type::new(&db, TypeKind::Int);
        let effect = EffectRow::open(&db, row_var);
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: int_ty,
                effect,
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let continuation_ty = Type::new(
            &db,
            TypeKind::Continuation {
                arg: int_ty,
                result: int_ty,
                effect,
            },
        );

        assert!(solver.occurs_in(*var_id, func_ty));
        assert!(solver.occurs_in(*var_id, continuation_ty));
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
                ability_id: test_ability_id(&db, "State"),
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
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let func2 = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect: empty_effect,
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                minimum_convention: crate::ast::CallingConvention::Direct,
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
            ability_id: test_ability_id(&db, "State"),
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
                id: TypeDefId::builtin_list(&db),
                name: trunk_ir::Symbol::new("List"),
                args: vec![var_ty],
            },
        );
        let list_int = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::builtin_list(&db),
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
                id: TypeDefId::builtin_list(&db),
                name: trunk_ir::Symbol::new("List"),
                args: vec![int_ty],
            },
        );
        let option_int = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Option")),
                name: trunk_ir::Symbol::new("Option"),
                args: vec![int_ty],
            },
        );

        let result = solver.unify_types(list_int, option_int);
        assert!(matches!(result, Err(SolveError::TypeMismatch { .. })));
    }

    #[test]
    fn test_unify_named_types_rejects_builtin_and_source_with_same_spelling() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);
        let name = Symbol::new("List");
        let int_ty = Type::new(&db, TypeKind::Int);
        let builtin = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::builtin_list(&db),
                name,
                args: vec![int_ty],
            },
        );
        let source = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::source(&db, name, crate::ast::NodeId::from_raw(1)),
                name,
                args: vec![int_ty],
            },
        );

        assert!(matches!(
            solver.unify_types(builtin, source),
            Err(SolveError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn test_unify_named_types_rejects_same_spelled_source_declarations() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);
        let name = Symbol::new("Thing");
        let int_ty = Type::new(&db, TypeKind::Int);
        let first = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::source(
                    &db,
                    Symbol::new("A::Thing"),
                    crate::ast::NodeId::from_raw(1),
                ),
                name,
                args: vec![int_ty],
            },
        );
        let second = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::source(
                    &db,
                    Symbol::new("B::Thing"),
                    crate::ast::NodeId::from_raw(2),
                ),
                name,
                args: vec![int_ty],
            },
        );

        assert!(matches!(
            solver.unify_types(first, second),
            Err(SolveError::TypeMismatch { .. })
        ));
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
                id: TypeDefId::builtin_list(&db),
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
    fn test_never_equality_is_strict() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let never_ty = Type::new(&db, TypeKind::Never);
        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let nat_ty = Type::new(&db, TypeKind::Nat);

        assert!(solver.unify_types(never_ty, never_ty).is_ok());
        assert!(solver.unify_types(never_ty, int_ty).is_err());
        assert!(solver.unify_types(bool_ty, never_ty).is_err());
        assert!(solver.unify_types(never_ty, nat_ty).is_err());

        // UniVar should unify with Never, then resolve to Never
        let var = fresh_var(&db, 0);
        assert!(solver.unify_types(var, never_ty).is_ok());
        assert_eq!(solver.type_subst.apply(&db, var), never_ty);

        // A variable resolved to Never retains that exact identity.
        let var2 = fresh_var(&db, 1);
        assert!(solver.unify_types(var2, never_ty).is_ok());
        assert!(solver.unify_types(var2, int_ty).is_err());
    }

    #[test]
    fn test_never_in_named_type_args() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);

        let never_ty = Type::new(&db, TypeKind::Never);
        let int_ty = Type::new(&db, TypeKind::Int);

        // Expression elimination does not recurse into nominal arguments.
        let list_never = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::builtin_list(&db),
                name: Symbol::new("List"),
                args: vec![never_ty],
            },
        );
        let list_int = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::builtin_list(&db),
                name: Symbol::new("List"),
                args: vec![int_ty],
            },
        );
        assert!(solver.unify_types(list_never, list_int).is_err());
    }

    #[test]
    fn test_never_coercion_preserves_expected_variable() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);
        let never = Type::new(&db, TypeKind::Never);
        let expected = fresh_var(&db, 0);
        let origin = ConstraintOrigin {
            node_id: crate::ast::NodeId::from_raw(0),
            kind: super::super::constraint::ConstraintOriginKind::Expression,
        };
        let mut constraints = ConstraintSet::new();
        constraints.add_type_coerce(never, expected, origin);
        solver.solve(constraints).unwrap();
        solver.finalize_relations().unwrap();
        assert_eq!(solver.type_subst.apply(&db, expected), expected);
        solver
            .unify_types(expected, Type::new(&db, TypeKind::Nat))
            .unwrap();
    }

    #[test]
    fn deferred_producers_with_equal_results_retain_distinct_dependencies() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);
        let first = fresh_var(&db, 0);
        let second = fresh_var(&db, 1);
        let result = Type::new(&db, TypeKind::Bool);
        let first_node = NodeId::from_raw(1);
        let second_node = NodeId::from_raw(2);
        solver.defer_producer(first_node, result, vec![first]);
        solver.defer_producer(second_node, result, vec![second]);
        assert_eq!(solver.pending_variables().0.len(), 2);
        solver.resolve_producer(first_node);
        let TypeKind::UniVar { id } = second.kind(&db) else {
            unreachable!();
        };
        assert_eq!(solver.pending_variables().0, vec![*id]);
    }

    #[test]
    fn deferred_join_keeps_actual_producer_and_reports_a_late_mismatch_once() {
        let db = test_db();
        let mut solver = TypeSolver::new(&db);
        let actual = fresh_var(&db, 0);
        let result = fresh_var(&db, 1);
        let nat = Type::new(&db, TypeKind::Nat);
        let origin = ConstraintOrigin {
            node_id: NodeId::from_raw(1),
            kind: super::super::constraint::ConstraintOriginKind::Expression,
        };
        solver.defer_producer(origin.node_id, actual, vec![]);
        let mut constraints = ConstraintSet::new();
        constraints.add(Constraint::TypeJoin {
            sources: vec![(actual, origin), (nat, origin)],
            result,
            origin,
            complete: true,
        });
        constraints.add_type_coerce(result, nat, origin);
        solver.solve(constraints).unwrap();
        solver.finalize_relations().unwrap();
        assert_eq!(solver.type_subst.apply(&db, actual), actual);
        assert_eq!(solver.type_subst.apply(&db, result), nat);
        solver.resolve_producer(origin.node_id);
        let mut late = ConstraintSet::new();
        late.add_type_eq(actual, Type::new(&db, TypeKind::Bool));
        let failure = solver.solve_with_origin(late).unwrap_err();
        assert_eq!(failure.origin, Some(origin));
        assert!(matches!(failure.error, SolveError::TypeMismatch { .. }));
        solver.finalize_relations().unwrap();
        solver.solve(ConstraintSet::new()).unwrap();
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
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        // Apply both substitutions
        let result = type_subst.apply_with_rows(&db, func_ty, &row_subst);

        // Check params and result are substituted
        if let TypeKind::Func {
            params,
            result,
            effect,
            ..
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
                ability_id: test_ability_id(&db, "State"),
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
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        let row_subst = RowSubst::new();
        let result = type_subst.apply_with_rows(&db, func_ty, &row_subst);

        if let TypeKind::Func { effect, .. } = result.kind(&db) {
            let effects = effect.effects(&db);
            assert_eq!(effects.len(), 1);
            assert_eq!(
                effects[0].ability_id.name(&db),
                trunk_ir::Symbol::new("State")
            );
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
                ability_id: test_ability_id(&db, "State"),
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
                minimum_convention: crate::ast::CallingConvention::Direct,
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
    #[should_panic(
        expected = "quantified type variable (BoundVar or LocalBoundVar) reached solver"
    )]
    fn test_bound_var_panics_in_debug() {
        // BoundVar and LocalBoundVar should never reach the solver — they must be instantiated first.
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
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                minimum_convention: crate::ast::CallingConvention::Direct,
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
            ability_id: test_ability_id(&db, "Console"),
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
            ability_id: test_ability_id(&db, "Console"),
            args: vec![],
        };
        let io = Effect {
            ability_id: test_ability_id(&db, "IO"),
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
            ability_id: test_ability_id(&db, "Console"),
            args: vec![],
        };
        let io = Effect {
            ability_id: test_ability_id(&db, "IO"),
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
        assert_eq!(effects[0].ability_id.name(&db), trunk_ir::Symbol::new("IO"));
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
            ability_id: test_ability_id(&db, "Console"),
            args: vec![],
        };
        let io = Effect {
            ability_id: test_ability_id(&db, "IO"),
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
        assert_eq!(
            e1_effects[0].ability_id.name(&db),
            trunk_ir::Symbol::new("IO")
        );
        assert!(e1_resolved.rest(&db).is_some()); // Open with e3

        // e2 should be bound to {Console | e3}
        let e2_resolved = solver.row_subst.get(e2.id).unwrap();
        let e2_effects = e2_resolved.effects(&db);
        assert_eq!(e2_effects.len(), 1);
        assert_eq!(
            e2_effects[0].ability_id.name(&db),
            trunk_ir::Symbol::new("Console")
        );
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
            ability_id: test_ability_id(&db, "State"),
            args: vec![var_ty],
        };
        let state_int = Effect {
            ability_id: test_ability_id(&db, "State"),
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
            ability_id: test_ability_id(&db, "Console"),
            args: vec![],
        };
        let io = Effect {
            ability_id: test_ability_id(&db, "IO"),
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
            ability_id: test_ability_id(&db, "State"),
            args: vec![int_ty],
        };
        let state_no_arg = Effect {
            ability_id: test_ability_id(&db, "State"),
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
            ability_id: test_ability_id(&db, "State"),
            args: vec![int_ty],
        };
        let state_bool = Effect {
            ability_id: test_ability_id(&db, "State"),
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
            ability_id: test_ability_id(&db, "State"),
            args: vec![int_ty],
        };
        let state_int2 = Effect {
            ability_id: test_ability_id(&db, "State"),
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
            ability_id: test_ability_id(&db, "State"),
            args: vec![var_ty],
        };
        let state_int = Effect {
            ability_id: test_ability_id(&db, "State"),
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
            ability_id: test_ability_id(&db, "State"),
            args: vec![var_a],
        };
        let state_b = Effect {
            ability_id: test_ability_id(&db, "State"),
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
            ability_id: test_ability_id(&db, "Pair"),
            args: vec![var_a, int_ty],
        };
        let pair2 = Effect {
            ability_id: test_ability_id(&db, "Pair"),
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

        // Compatibility uses equality, including for effect arguments.
        let never_ty = Type::new(&db, TypeKind::Never);
        assert!(!solver.types_unifiable(never_ty, int_ty));
        assert!(!solver.types_unifiable(bool_ty, never_ty));
        assert!(solver.types_unifiable(never_ty, var_ty));
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
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let func2 = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect,
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                id: TypeDefId::builtin_list(&db),
                name: trunk_ir::Symbol::new("List"),
                args: vec![],
            },
        );
        let option_ctor = Type::new(
            &db,
            TypeKind::Named {
                id: TypeDefId::synthetic(&db, trunk_ir::Symbol::new("Option")),
                name: trunk_ir::Symbol::new("Option"),
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
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let outer_effect = EffectRow::new(&db, vec![], None);
        let outer_func = Type::new(
            &db,
            TypeKind::Func {
                params: vec![inner_func],
                result: int_ty,
                effect: outer_effect,
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        let outer_effect = EffectRow::new(&db, vec![], None);
        let outer_func = Type::new(
            &db,
            TypeKind::Func {
                params: vec![],
                result: inner_func,
                effect: outer_effect,
                minimum_convention: crate::ast::CallingConvention::Direct,
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
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );

        assert!(
            !solver.row_occurs_in_type(row_var, func),
            "Row variable not present should not be detected"
        );
    }

    #[test]
    fn test_types_unifiable_continuation() {
        let db = test_db();
        let solver = TypeSolver::new(&db);

        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);

        let pure = EffectRow::pure(&db);

        // Same continuation types should be unifiable
        let cont1 = Type::new(
            &db,
            TypeKind::Continuation {
                arg: int_ty,
                result: bool_ty,
                effect: pure,
            },
        );
        let cont2 = Type::new(
            &db,
            TypeKind::Continuation {
                arg: int_ty,
                result: bool_ty,
                effect: pure,
            },
        );
        assert!(solver.types_unifiable(cont1, cont2));

        // Different arg types should not be unifiable
        let cont3 = Type::new(
            &db,
            TypeKind::Continuation {
                arg: bool_ty,
                result: bool_ty,
                effect: pure,
            },
        );
        assert!(!solver.types_unifiable(cont1, cont3));

        // Different result types should not be unifiable
        let cont4 = Type::new(
            &db,
            TypeKind::Continuation {
                arg: int_ty,
                result: int_ty,
                effect: pure,
            },
        );
        assert!(!solver.types_unifiable(cont1, cont4));
    }
}

#[cfg(test)]
mod row_union_tests {
    use super::*;
    use salsa_test_macros::salsa_test;

    #[salsa_test]
    fn row_removal_defers_ambiguous_types(db: &salsa::DatabaseImpl) {
        let ability = crate::ast::AbilityId::source(db, trunk_ir::Symbol::new("Writer"));
        let nat = Type::new(db, TypeKind::Nat);
        let int = Type::new(db, TypeKind::Int);
        let row = |ty| {
            EffectRow::single(
                db,
                Effect {
                    ability_id: ability,
                    args: vec![ty],
                },
            )
        };
        for reverse in [false, true] {
            let mut solver = TypeSolver::new(db);
            let result = EffectRow::open(db, solver.fresh_row_var());
            let mut labels = vec![
                row(nat).effects(db)[0].clone(),
                row(int).effects(db)[0].clone(),
            ];
            if reverse {
                labels.reverse();
            }
            solver.add_row_removals(vec![crate::ast::RowRemoval {
                source: EffectRow::new(db, labels, None),
                removed: row(nat),
                result,
            }]);
            solver.finalize_relations().unwrap();
            assert_eq!(solver.row_subst().apply(db, result), row(int));
        }
        let mut solver = TypeSolver::new(db);
        let unknown = solver.fresh_type_var(db);
        let result = EffectRow::open(db, solver.fresh_row_var());
        solver.add_row_removals(vec![crate::ast::RowRemoval {
            source: row(unknown),
            removed: row(nat),
            result,
        }]);
        solver.finalize_relations().unwrap();
        assert_eq!(solver.type_subst().apply(db, unknown), unknown);
        assert_eq!(solver.retained_row_removals().len(), 1);
        let mut constraints = ConstraintSet::new();
        constraints.add_type_eq(unknown, int);
        solver.solve(constraints).unwrap();
        assert_eq!(solver.row_subst().apply(db, result), row(int));
        assert!(solver.retained_row_removals().is_empty());
    }

    #[salsa_test]
    fn row_removal_empty_result_does_not_close_its_source(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);
        let source = EffectRow::open(db, solver.fresh_row_var());
        solver.add_row_removals(vec![crate::ast::RowRemoval {
            source,
            removed: label(db, "Ping"),
            result: EffectRow::pure(db),
        }]);
        solver.finalize_relations().unwrap();
        assert_eq!(solver.row_subst().apply(db, source), source);
        let mut constraints = ConstraintSet::new();
        constraints.add_row_eq(source, label(db, "Ping"));
        solver.solve(constraints).unwrap();
        assert!(solver.retained_row_removals().is_empty());
    }

    #[salsa_test]
    fn row_removal_rejects_reintroduced_labels(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);
        let source = EffectRow::open(db, solver.fresh_row_var());
        solver.add_row_removals(vec![crate::ast::RowRemoval {
            source,
            removed: label(db, "Ping"),
            result: label(db, "Ping"),
        }]);
        assert!(solver.finalize_relations().is_err());
    }

    #[salsa_test]
    fn row_removal_uses_exact_ability_identity(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);
        let name = trunk_ir::Symbol::new("Io");
        let builtin = crate::ast::AbilityId::new(
            db,
            crate::ast::AbilityOrigin::Builtin(crate::ast::BuiltinAbility::Io),
            name,
        );
        let source = crate::ast::AbilityId::source(db, name);
        let effect = |ability_id| Effect {
            ability_id,
            args: vec![],
        };
        let result = EffectRow::open(db, solver.fresh_row_var());
        solver.add_row_removals(vec![crate::ast::RowRemoval {
            source: EffectRow::new(db, vec![effect(builtin), effect(source)], None),
            removed: EffectRow::single(db, effect(builtin)),
            result,
        }]);
        solver.finalize_relations().unwrap();
        assert_eq!(
            solver.row_subst().apply(db, result),
            EffectRow::single(db, effect(source))
        );
    }

    fn label<'db>(db: &'db dyn salsa::Database, name: &str) -> EffectRow<'db> {
        EffectRow::single(
            db,
            Effect {
                ability_id: crate::ast::AbilityId::source(db, trunk_ir::Symbol::from_dynamic(name)),
                args: vec![],
            },
        )
    }

    #[salsa_test]
    fn unrelated_row_union_stays_in_solver_without_entering_value_scheme(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);
        let left = EffectRow::open(db, EffectVar { id: 1 });
        let right = EffectRow::open(db, EffectVar { id: 2 });
        solver.add_row_unions(vec![crate::ast::RowUnion {
            sources: vec![left, right],
            result: label(db, "Writer"),
        }]);
        solver.finalize_relations().unwrap();
        assert!(
            solver
                .row_unions_for_type(Type::new(db, TypeKind::Nat))
                .is_empty()
        );
        assert_eq!(solver.retained_row_unions().len(), 1);
        let mut constraints = ConstraintSet::new();
        constraints.add_row_eq(left, label(db, "Reader"));
        assert!(solver.solve(constraints).is_err());
    }

    #[salsa_test]
    fn scheme_row_dependencies_follow_shared_effect_type_arguments(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);
        let variable = Type::new(
            db,
            TypeKind::UniVar {
                id: UniVarId::new(db, crate::ast::UniVarSource::Anonymous(813), 0),
            },
        );
        let result = EffectRow::single(
            db,
            Effect {
                ability_id: crate::ast::AbilityId::source(db, trunk_ir::Symbol::new("Writer")),
                args: vec![variable],
            },
        );
        for first in [1, 3] {
            solver.add_row_unions(vec![crate::ast::RowUnion {
                sources: vec![
                    EffectRow::open(db, EffectVar { id: first }),
                    EffectRow::open(db, EffectVar { id: first + 1 }),
                ],
                result,
            }]);
        }
        solver.finalize_relations().unwrap();
        let nil = Type::new(db, TypeKind::Nil);
        let callable = Type::new(
            db,
            TypeKind::Func {
                params: vec![],
                result: nil,
                effect: EffectRow::open(db, EffectVar { id: 1 }),
                minimum_convention: crate::ast::CallingConvention::Direct,
            },
        );
        // The second union is reachable only through the first union's
        // Writer argument, not through a shared row variable.
        assert_eq!(solver.row_unions_for_type(callable).len(), 2);
        assert_eq!(solver.row_unions_for_type(variable).len(), 2);
    }

    #[salsa_test]
    fn row_union_retains_both_delayed_sources(db: &salsa::DatabaseImpl) {
        for reverse in [false, true] {
            let mut solver = TypeSolver::new(db);
            let left = EffectRow::open(db, EffectVar { id: 10 });
            let right = EffectRow::open(db, EffectVar { id: 11 });
            let result = EffectRow::open(db, EffectVar { id: 12 });
            let sources = if reverse {
                vec![right, left]
            } else {
                vec![left, right]
            };
            solver.add_row_unions(vec![crate::ast::RowUnion { sources, result }]);
            solver.finalize_relations().unwrap();
            assert_eq!(solver.retained_row_unions().len(), 1);
            assert!(solver.row_subst.get(10).is_none());
            assert!(solver.row_subst.get(11).is_none());
            let mut constraints = ConstraintSet::new();
            constraints.add_row_eq(left, label(db, "Reader"));
            solver.solve(constraints).unwrap();
            let mut constraints = ConstraintSet::new();
            constraints.add_row_eq(right, label(db, "Writer"));
            solver.solve(constraints).unwrap();
            let result = solver.row_subst.apply(db, result);
            assert!(result.rest(db).is_none());
            assert_eq!(result.effects(db).len(), 2);
            assert!(solver.retained_row_unions().is_empty());
        }
    }

    #[salsa_test]
    fn row_union_closed_result_does_not_choose_a_source(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);
        let left = EffectRow::open(db, EffectVar { id: 1 });
        let right = EffectRow::open(db, EffectVar { id: 2 });
        solver.add_row_unions(vec![crate::ast::RowUnion {
            sources: vec![left, right],
            result: label(db, "Writer"),
        }]);
        solver.finalize_relations().unwrap();
        assert!(solver.row_subst.get(1).is_none());
        assert!(solver.row_subst.get(2).is_none());
        let mut constraints = ConstraintSet::new();
        constraints.add_row_eq(left, label(db, "Reader"));
        assert!(solver.solve(constraints).is_err());
    }

    #[salsa_test]
    fn row_union_pure_result_closes_every_source(db: &salsa::DatabaseImpl) {
        let mut solver = TypeSolver::new(db);
        let left = EffectRow::open(db, EffectVar { id: 1 });
        let right = EffectRow::open(db, EffectVar { id: 2 });
        solver.add_row_unions(vec![crate::ast::RowUnion {
            sources: vec![left, right],
            result: EffectRow::pure(db),
        }]);
        solver.finalize_relations().unwrap();
        assert!(solver.row_subst.apply(db, left).is_pure(db));
        assert!(solver.row_subst.apply(db, right).is_pure(db));
    }
    #[salsa_test]
    fn row_union_defers_ambiguous_instance_until_type_constraint_arrives(db: &salsa::DatabaseImpl) {
        let variable = Type::new(
            db,
            TypeKind::UniVar {
                id: UniVarId::new(db, crate::ast::UniVarSource::Anonymous(812), 0),
            },
        );
        let nat = Type::new(db, TypeKind::Nat);
        let int = Type::new(db, TypeKind::Int);
        let ability = crate::ast::AbilityId::source(db, trunk_ir::Symbol::new("Writer"));
        let effect = |arg| Effect {
            ability_id: ability,
            args: vec![arg],
        };
        for (reverse, swap) in [(false, false), (true, false), (false, true), (true, true)] {
            let source = EffectRow::new(db, vec![effect(variable), effect(nat)], None);
            let result = EffectRow::new(
                db,
                if reverse {
                    vec![effect(int), effect(nat)]
                } else {
                    vec![effect(nat), effect(int)]
                },
                None,
            );
            let mut solver = TypeSolver::new(db);
            let mut constraints = ConstraintSet::new();
            if swap {
                constraints.add_row_eq(result, source);
            } else {
                constraints.add_row_eq(source, result);
            }
            solver.solve(constraints).unwrap();
            assert_eq!(solver.type_subst().apply(db, variable), variable);
            assert!(!solver.retained_row_unions().is_empty());
            let mut constraints = ConstraintSet::new();
            constraints.add_type_eq(variable, int);
            solver.solve(constraints).unwrap();
            solver.finalize_relations().unwrap();
            assert!(solver.retained_row_unions().is_empty());
        }
    }
}
