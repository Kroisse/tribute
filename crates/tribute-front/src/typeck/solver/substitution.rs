//! Solver substitutions and binder-aware generalization.

use super::{
    EffectRow, HashMap, RowSubst, SmallVec, Type, TypeKind, TypeParam, TypeSubst, UniVarId,
    map_effect_row_type_args,
};

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
        let (generalized, type_params, _) = self.generalize_with_mapping(db, ty, row_subst);
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
        self.collect_univars_from_type(db, ty, row_subst, &mut univars);
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
        self.collect_univars_from_type(db, ty, row_subst, &mut univars);

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
    pub fn collect_univars_from_type(
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
                    self.collect_univars_from_type(db, subst_ty, row_subst, out);
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
                    self.collect_univars_from_type(db, *p, row_subst, out);
                }
                self.collect_univars_from_type(db, *result, row_subst, out);
                self.collect_univars_from_effect_row(db, *effect, row_subst, out);
            }
            TypeKind::Named { args, .. } => {
                for a in args {
                    self.collect_univars_from_type(db, *a, row_subst, out);
                }
            }
            TypeKind::Tuple(elems) => {
                for e in elems {
                    self.collect_univars_from_type(db, *e, row_subst, out);
                }
            }
            TypeKind::App { ctor, args } => {
                self.collect_univars_from_type(db, *ctor, row_subst, out);
                for a in args {
                    self.collect_univars_from_type(db, *a, row_subst, out);
                }
            }
            TypeKind::Continuation {
                arg,
                result,
                effect,
            } => {
                self.collect_univars_from_type(db, *arg, row_subst, out);
                self.collect_univars_from_type(db, *result, row_subst, out);
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
                self.collect_univars_from_type(db, *a, row_subst, out);
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
