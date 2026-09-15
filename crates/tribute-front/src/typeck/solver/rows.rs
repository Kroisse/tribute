//! Effect-row equality, set matching, and retained relations.

use super::{
    EffectRow, EffectVar, HashMap, LocatedSolveError, SolveError, Type, TypeKind, TypeSolver,
    UniVarId, collect_effect_vars, map_effect_row_type_args,
};

impl<'db> TypeSolver<'db> {
    pub(super) fn union_variables(
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
    pub(super) fn row_relations_for_type(
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
    pub(super) fn row_dependency_views(&self) -> Vec<crate::ast::RowUnion<'db>> {
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

    pub(super) fn normalize_row(&self, row: EffectRow<'db>) -> EffectRow<'db> {
        let row = self.row_subst.apply(self.db, row);
        map_effect_row_type_args(self.db, row, |ty| {
            self.type_subst
                .apply_with_rows(self.db, ty, &self.row_subst)
        })
    }

    pub(super) fn settle_row_unions(&mut self) -> Result<(), LocatedSolveError<'db>> {
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

    pub(super) fn solve_row_removal(
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

    pub(super) fn solve_row_union(
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

    /// Apply type substitution to effect arguments in a row.
    ///
    /// This ensures that UniVars in effect args (like `State(s)` where `s` is a UniVar)
    /// are resolved before row comparison. Without this, two rows with the same ability
    /// but different (unresolved vs resolved) args would fail to unify.
    pub(super) fn apply_type_subst_to_row(&self, row: EffectRow<'db>) -> EffectRow<'db> {
        map_effect_row_type_args(self.db, row, |a| self.type_subst.apply(self.db, a))
    }

    /// Check if a row variable occurs in an effect row (for row occurs check).
    pub(super) fn row_occurs_in(&self, var: EffectVar, row: EffectRow<'db>) -> bool {
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
    pub(super) fn row_occurs_in_type(&self, var: EffectVar, ty: Type<'db>) -> bool {
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
    pub(super) fn unify_rows(
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

    pub(super) fn unify_rows_inner(
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
    pub(super) fn unify_effects_as_sets(
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

    /// Compute effect difference with unification support.
    ///
    /// Returns matched effects and unmatched effects from list2.
    pub(super) fn compute_effect_difference_with_unify(
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
    pub(super) fn compute_effect_split_with_unify(
        &mut self,
        list1: &[crate::ast::Effect<'db>],
        list2: &[crate::ast::Effect<'db>],
    ) -> Result<(Vec<crate::ast::Effect<'db>>, Vec<crate::ast::Effect<'db>>), SolveError<'db>> {
        let only_list1 = self.compute_effect_difference_with_unify(list2, list1)?;
        let only_list2 = self.compute_effect_difference_with_unify(list1, list2)?;
        Ok((only_list1, only_list2))
    }
}
