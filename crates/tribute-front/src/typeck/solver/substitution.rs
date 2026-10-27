//! Solver substitutions and binder-aware generalization.

use std::convert::Infallible;
use std::ops::ControlFlow;

use super::{
    AbilityRow, HashMap, RowSubst, RowVar, SmallVec, Type, TypeKind, TypeParam, TypeSubst, UniVarId,
};
use crate::ast::fold::{TypeFoldable, TypeFolder, TypeVisitor};

impl<'db> TypeSubst<'db> {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self {
            map: HashMap::default(),
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
    /// If you need to substitute ability row variables, use `apply_with_rows` directly
    /// with the appropriate `RowSubst`.
    pub fn apply(&self, db: &'db dyn salsa::Database, ty: Type<'db>) -> Type<'db> {
        self.apply_with_rows(db, ty, &RowSubst::new())
    }

    /// Apply substitution to a type, including ability row substitution.
    pub fn apply_with_rows(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
    ) -> Type<'db> {
        ty.fold_with(&mut Resolve {
            db,
            types: self,
            rows: row_subst,
            bound: &HashMap::default(),
            local: &HashMap::default(),
        })
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
            return (ty, Vec::new(), HashMap::default());
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
        ty.fold_with(&mut Resolve {
            db,
            types: self,
            rows: row_subst,
            bound: var_to_index,
            local: local_vars,
        })
    }

    /// Collect unresolved UniVarIds from a type in appearance (left-to-right) order.
    pub fn collect_univars_from_type(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
        out: &mut Vec<UniVarId<'db>>,
    ) {
        let ControlFlow::Continue(()) = ty.visit_with(&mut Unresolved {
            db,
            types: self,
            rows: row_subst,
            out,
        });
    }

    /// Replace unresolved UniVars with BoundVars according to the given mapping.
    fn replace_univars_with_bound(
        &self,
        db: &'db dyn salsa::Database,
        ty: Type<'db>,
        row_subst: &RowSubst<'db>,
        var_to_index: &HashMap<UniVarId<'db>, u32>,
    ) -> Type<'db> {
        self.replace_univars_with_bound_and_local(
            db,
            ty,
            row_subst,
            var_to_index,
            &HashMap::default(),
        )
    }
}

/// Resolves solver variables: a substituted `UniVar` is resolved again and
/// a row takes its row substitution before its arguments are resolved.
/// Generalization also maps variables to their function-scheme (`bound`) or
/// local (`local`) binders, local first. Bound variables of either kind pass
/// through.
struct Resolve<'a, 'db> {
    db: &'db dyn salsa::Database,
    types: &'a TypeSubst<'db>,
    rows: &'a RowSubst<'db>,
    bound: &'a HashMap<UniVarId<'db>, u32>,
    local: &'a HashMap<UniVarId<'db>, (crate::ast::NodeId, u32)>,
}

impl<'db> TypeFolder<'db> for Resolve<'_, 'db> {
    type Error = Infallible;

    fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    fn fold_type(&mut self, ty: Type<'db>) -> Result<Type<'db>, Infallible> {
        let TypeKind::UniVar { id } = ty.kind(self.db) else {
            return ty.try_super_fold_with(self);
        };
        Ok(if let Some(&(scope, index)) = self.local.get(id) {
            Type::new(self.db, TypeKind::LocalBoundVar { scope, index })
        } else if let Some(&index) = self.bound.get(id) {
            Type::new(self.db, TypeKind::BoundVar { index })
        } else if let Some(resolved) = self.types.get(*id) {
            resolved.fold_with(self)
        } else {
            ty
        })
    }

    fn fold_row(&mut self, row: AbilityRow<'db>) -> Result<AbilityRow<'db>, Infallible> {
        self.rows.apply(self.db, row).try_super_fold_with(self)
    }
}

/// Collects unresolved `UniVar`s, looking through the solver substitutions as
/// [`Resolve`] does, in first appearance order.
struct Unresolved<'a, 'db> {
    db: &'db dyn salsa::Database,
    types: &'a TypeSubst<'db>,
    rows: &'a RowSubst<'db>,
    out: &'a mut Vec<UniVarId<'db>>,
}

impl<'db> TypeVisitor<'db> for Unresolved<'_, 'db> {
    type Break = Infallible;

    fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    fn visit_type(&mut self, ty: Type<'db>) -> ControlFlow<Infallible> {
        let TypeKind::UniVar { id } = ty.kind(self.db) else {
            return ty.super_visit_with(self);
        };
        if let Some(resolved) = self.types.get(*id) {
            self.visit_type(resolved)?;
        } else if !self.out.contains(id) {
            self.out.push(*id);
        }
        ControlFlow::Continue(())
    }

    fn visit_row(&mut self, row: AbilityRow<'db>) -> ControlFlow<Infallible> {
        self.rows.apply(self.db, row).super_visit_with(self)
    }
}

impl<'db> RowSubst<'db> {
    /// Create an empty substitution.
    pub fn new() -> Self {
        Self {
            map: HashMap::default(),
        }
    }

    /// Insert a mapping.
    pub fn insert(&mut self, var: u64, row: AbilityRow<'db>) {
        self.map.insert(var, row);
    }

    /// Look up a row variable.
    pub fn get(&self, var: u64) -> Option<AbilityRow<'db>> {
        self.map.get(&var).copied()
    }

    /// Make `var` the representative of the bare open row it resolves to,
    /// so finalized types name `var` instead of the solver's alias. A row
    /// that resolved to concrete effects or no tail is left unchanged.
    pub(crate) fn make_representative(&mut self, db: &'db dyn salsa::Database, var: RowVar) {
        let resolved = self.apply(db, AbilityRow::open(db, var));
        if let Some(alias) = resolved.rest(db)
            && alias != var
            && resolved.instances(db).is_empty()
        {
            self.map.remove(&var.id);
            self.map.insert(alias.id, AbilityRow::open(db, var));
        }
    }

    /// Apply substitution to an ability row.
    pub fn apply(&self, db: &'db dyn salsa::Database, row: AbilityRow<'db>) -> AbilityRow<'db> {
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

            let mut effects = row.instances(db).to_vec();
            for effect in subst_row.instances(db) {
                if !effects.contains(effect) {
                    effects.push(effect.clone());
                }
            }
            let rest = subst_row.rest(db);
            row = AbilityRow::new(db, effects, rest);
        }
        row
    }
}
