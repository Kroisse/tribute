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

    pub(crate) fn normalize_row(&self, row: EffectRow<'db>) -> EffectRow<'db> {
        let row = self.row_subst.apply(self.db, row);
        map_effect_row_type_args(self.db, row, |ty| {
            self.type_subst
                .apply_with_rows(self.db, ty, &self.row_subst)
        })
    }

    pub(super) fn settle_row_unions(&mut self) -> Result<(), LocatedSolveError<'db>> {
        let nested = std::mem::replace(&mut self.solving_row_relation, true);
        let result = self.settle_row_unions_inner();
        self.solving_row_relation = nested;
        result
    }

    fn settle_row_unions_inner(&mut self) -> Result<(), LocatedSolveError<'db>> {
        let mut first_error = None;
        for (union, origin) in std::mem::take(&mut self.pending_row_unions) {
            // A union deferred again keeps the origin of its constraint.
            let outer = std::mem::replace(&mut self.current_origin, origin);
            let solved = self.solve_row_union(&union);
            self.current_origin = outer;
            match solved {
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

    /// Discharge the relations a function body left over its signature rows.
    ///
    /// The signature rows are rigid, so a relation that only defines a
    /// body-local row from them has one solution:
    ///
    /// - Removing labels the source row names explicitly leaves its tail
    ///   untouched, since a row holds each label once.
    /// - Joining rows whose tails a declared union covers yields that
    ///   union's result.
    /// - A body-local tail joined into a closed row, with nothing else to
    ///   constrain it, is empty.
    ///
    /// Relations that would constrain the signature rows themselves remain
    /// pending for the caller to report.
    pub(crate) fn settle_signature_relations(
        &mut self,
        declared_unions: &[crate::ast::RowUnion<'db>],
    ) -> Result<(), LocatedSolveError<'db>> {
        let nested = std::mem::replace(&mut self.solving_row_relation, true);
        let result = self.settle_signature_relations_inner(declared_unions);
        self.solving_row_relation = nested;
        result
    }

    fn settle_signature_relations_inner(
        &mut self,
        declared_unions: &[crate::ast::RowUnion<'db>],
    ) -> Result<(), LocatedSolveError<'db>> {
        let declared: Vec<_> = declared_unions
            .iter()
            .filter_map(|union| {
                let result = self.normalize_row(union.result).rest(self.db)?;
                let sources = union
                    .sources
                    .iter()
                    .filter_map(|row| self.normalize_row(*row).rest(self.db))
                    .collect::<Vec<_>>();
                Some((result, sources))
            })
            .collect();
        loop {
            let before = self.pending_row_unions.len() + self.pending_row_removals.len();
            let mut bound = 0;
            for (removal, origin) in std::mem::take(&mut self.pending_row_removals) {
                let mut normalized = removal.clone();
                normalized.for_each_row_mut(|row| *row = self.normalize_row(*row));
                let source_effects = normalized.source.effects(self.db);
                match normalized.source.rest(self.db) {
                    Some(tail)
                        if normalized
                            .removed
                            .effects(self.db)
                            .iter()
                            .all(|effect| source_effects.contains(effect)) =>
                    {
                        let remaining: Vec<_> = source_effects
                            .iter()
                            .filter(|effect| !normalized.removed.effects(self.db).contains(effect))
                            .cloned()
                            .collect();
                        let row = EffectRow::new(self.db, remaining, Some(tail));
                        self.unify_rows(normalized.result, row)
                            .map_err(|error| LocatedSolveError { error, origin })?;
                    }
                    _ => self.pending_row_removals.push((removal, origin)),
                }
            }
            for (union, origin) in std::mem::take(&mut self.pending_row_unions) {
                let mut normalized = union.clone();
                normalized.for_each_row_mut(|row| *row = self.normalize_row(*row));
                let mut known = Vec::new();
                let mut tails = Vec::new();
                for source in &normalized.sources {
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
                let cover = declared.iter().find(|(result, sources)| {
                    let covered = tails
                        .iter()
                        .all(|tail| tail == result || sources.contains(tail));
                    let complete =
                        tails.contains(result) || sources.iter().all(|row| tails.contains(row));
                    tails.len() > 1 && covered && complete
                });
                match cover {
                    Some((result, _)) => {
                        let row = EffectRow::new(self.db, known, Some(*result));
                        self.unify_rows(normalized.result, row)
                            .map_err(|error| LocatedSolveError { error, origin })?;
                    }
                    _ => {
                        // A body-local tail that nothing else constrains
                        // and that is joined with one signature row adds no
                        // labels of its own: it is that row.
                        let (free, fixed): (Vec<_>, Vec<_>) = tails
                            .into_iter()
                            .partition(|tail| self.is_unconstrained_tail(*tail, &normalized));
                        if normalized.result.rest(self.db).is_none() {
                            // A closed result bounds every source, and a
                            // body-local tail that nothing else constrains
                            // adds no labels of its own: it is empty.
                            for free in free {
                                self.row_subst.insert(free.id, EffectRow::pure(self.db));
                                bound += 1;
                            }
                        } else if let [tail] = fixed[..]
                            && self.rigid_rows.contains(&tail)
                        {
                            for free in free {
                                self.row_subst
                                    .insert(free.id, EffectRow::open(self.db, tail));
                                bound += 1;
                            }
                        }
                        self.pending_row_unions.push((union, origin));
                    }
                }
            }
            self.settle_row_unions()?;
            if bound == 0
                && self.pending_row_unions.len() + self.pending_row_removals.len() == before
            {
                return Ok(());
            }
        }
    }

    /// Whether `tail`, a source tail of `union`, is a body-local row that no
    /// signature row or other relation determines.
    fn is_unconstrained_tail(&self, tail: EffectVar, union: &crate::ast::RowUnion<'db>) -> bool {
        let is_tail = |row: EffectRow<'db>| self.normalize_row(row).rest(self.db) == Some(tail);
        !self.rigid_rows.contains(&tail)
            && union.result.rest(self.db) != Some(tail)
            && !self
                .pending_row_unions
                .iter()
                .any(|(other, _)| is_tail(other.result))
            && !self
                .pending_row_removals
                .iter()
                .any(|(removal, _)| is_tail(removal.result))
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

    /// Unify a union's sources with its result. The sources come first: a
    /// pure result must not take the call-site shortcut that lets a pure row
    /// ignore the effects of the other side. The result stays the expected
    /// side of a mismatch.
    fn unify_union_source(
        &mut self,
        sources: EffectRow<'db>,
        result: EffectRow<'db>,
    ) -> Result<(), SolveError<'db>> {
        self.unify_rows(sources, result)
            .map_err(|error| match error {
                SolveError::RowMismatch { expected, actual } => SolveError::RowMismatch {
                    expected: actual,
                    actual: expected,
                },
                error => error,
            })
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
            self.unify_union_source(actual, union.result)?;
            return Ok(true);
        }
        if union.result.is_pure(self.db) {
            for source in union.sources {
                self.unify_union_source(source, union.result)?;
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
        let row = map_effect_row_type_args(self.db, row, |a| self.type_subst.apply(self.db, a));
        // Instances that the substitution made equal are one instance.
        let effects = row.effects(self.db);
        let mut unique: Vec<crate::ast::Effect<'db>> = Vec::with_capacity(effects.len());
        for effect in effects {
            if !unique.contains(effect) {
                unique.push(effect.clone());
            }
        }
        if unique.len() == effects.len() {
            row
        } else {
            EffectRow::new(self.db, unique, row.rest(self.db))
        }
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
        if !self.solving_row_relation && self.waits_for_union(left, right) {
            self.pending_row_eqs
                .push((left, right, self.current_origin));
            return Ok(());
        }
        match self.unify_rows_inner(left, right) {
            Err(SolveError::AmbiguousEffect { .. }) => {
                let union = crate::ast::RowUnion {
                    sources: vec![left],
                    result: right,
                };
                if !self.pending_row_unions.iter().any(|(old, _)| *old == union) {
                    self.pending_row_unions.push((union, self.current_origin));
                }
                Ok(())
            }
            result => result,
        }
    }

    /// The first effect equality that stayed ambiguous after every relation
    /// settled.
    ///
    /// A union whose rows are all closed settles as soon as one instance
    /// matches each label, so one still pending is an equality deferred for
    /// ambiguity that no later type decided.
    pub(crate) fn unsettled_ambiguity(&self) -> Option<LocatedSolveError<'db>> {
        self.pending_row_unions.iter().find_map(|(union, origin)| {
            let result = self.normalize_row(union.result);
            let [source] = union.sources.as_slice() else {
                return None;
            };
            let source = self.normalize_row(*source);
            (result.rest(self.db).is_none() && source.rest(self.db).is_none()).then(|| {
                LocatedSolveError {
                    // The equality's left row is the expected side, as in
                    // `unify_rows`.
                    error: SolveError::AmbiguousEffect {
                        expected: source,
                        actual: result,
                    },
                    origin: *origin,
                }
            })
        })
    }

    /// Whether equating the rows must wait for a pending union.
    ///
    /// Equating `{A | u}` with a row that names labels binds `u` to what is
    /// left once the common labels are matched, as if `u` held none of them.
    /// While `u` is the result of a pending union, the union may still add
    /// `A` to it; the substituted row then holds `A` once, and the labels left
    /// for the other side differ.
    fn waits_for_union(&self, left: EffectRow<'db>, right: EffectRow<'db>) -> bool {
        if self.pending_row_unions.is_empty() {
            return false;
        }
        let left = self.normalize_row(left);
        let right = self.normalize_row(right);
        [(left, right), (right, left)]
            .into_iter()
            .any(|(row, other)| {
                let Some(tail) = row.rest(self.db) else {
                    return false;
                };
                let names_labels = |row: EffectRow<'db>| !row.effects(self.db).is_empty();
                names_labels(row)
                    && (names_labels(other) || other.rest(self.db).is_none())
                    && self.pending_row_unions.iter().any(|(union, _)| {
                        self.normalize_row(union.result).rest(self.db) == Some(tail)
                    })
            })
    }

    /// Unify the row equalities that no longer wait for a union, or all of
    /// them when `force` is set.
    pub(super) fn settle_row_eqs(&mut self, force: bool) -> Result<(), LocatedSolveError<'db>> {
        let mut first_error = None;
        for (left, right, origin) in std::mem::take(&mut self.pending_row_eqs) {
            if !force && self.waits_for_union(left, right) {
                self.pending_row_eqs.push((left, right, origin));
                continue;
            }
            let nested = std::mem::replace(&mut self.solving_row_relation, true);
            let result = self.unify_rows(left, right);
            self.solving_row_relation = nested;
            if let Err(error) = result {
                first_error.get_or_insert(LocatedSolveError { error, origin });
            }
        }
        first_error.map_or(Ok(()), Err)
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
                // Compute difference: effects in r2 but not in r1
                // First check that all effects in r1 have matches in r2
                let mut pairs = Vec::new();
                let only_r2 = self.compute_effect_difference(effects1, effects2, &mut pairs)?;

                // r1's effects must all be in r2
                // (if any effect from r1 is in only_r2, it means no match was found)
                let missing = self.compute_effect_difference(effects2, effects1, &mut pairs)?;
                if !missing.is_empty() {
                    return Err(SolveError::RowMismatch {
                        expected: r1,
                        actual: r2,
                    });
                }
                self.unify_effect_args(pairs)?;

                // Unifying effect arguments may have bound the tail itself.
                if self.row_subst.get(var1.id).is_some() {
                    return self.unify_rows_inner(r1, r2);
                }
                // Bind var1 to remaining effects (closed)
                let remainder = EffectRow::new(self.db, only_r2, None);
                if self.row_occurs_in(var1, remainder) {
                    return Err(SolveError::RowMismatch {
                        expected: r1,
                        actual: r2,
                    });
                }
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

                // Compute difference: effects in r1 but not in r2
                let mut pairs = Vec::new();
                let only_r1 = self.compute_effect_difference(effects2, effects1, &mut pairs)?;

                // r2's effects must all be in r1
                let missing = self.compute_effect_difference(effects1, effects2, &mut pairs)?;
                if !missing.is_empty() {
                    return Err(SolveError::RowMismatch {
                        expected: r1,
                        actual: r2,
                    });
                }
                self.unify_effect_args(pairs)?;
                // Unifying effect arguments may have bound the tail itself.
                if self.row_subst.get(var2.id).is_some() {
                    return self.unify_rows_inner(r1, r2);
                }
                // Bind var2 to remaining effects (closed)
                let remainder = EffectRow::new(self.db, only_r1, None);
                if self.row_occurs_in(var2, remainder) {
                    return Err(SolveError::RowMismatch {
                        expected: r1,
                        actual: r2,
                    });
                }
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
                let (only_r1, only_r2) =
                    self.compute_effect_split_with_unify(effects1, effects2)?;

                // Unifying effect arguments may have bound a tail itself.
                if self.row_subst.get(v1.id).is_some() || self.row_subst.get(v2.id).is_some() {
                    return self.unify_rows_inner(r1, r2);
                }
                // Row occurs check: `v1` is bound to `only_r2` and `v2` to
                // `only_r1`, so a tail may not occur in its own remainder,
                // nor may each occur in the other's while the other's tail
                // occurs in its own, which would cycle through both bindings.
                let only_r1_row = EffectRow::new(self.db, only_r1.clone(), None);
                let only_r2_row = EffectRow::new(self.db, only_r2.clone(), None);
                if self.row_occurs_in(v1, only_r2_row)
                    || self.row_occurs_in(v2, only_r1_row)
                    || (self.row_occurs_in(v1, only_r1_row) && self.row_occurs_in(v2, only_r2_row))
                {
                    return Err(SolveError::RowMismatch {
                        expected: r1,
                        actual: r2,
                    });
                }

                // A signature row stays the common tail when the other side
                // adds no labels to it.
                let rigid1 = self.rigid_rows.contains(&v1);
                let rigid2 = self.rigid_rows.contains(&v2);
                if rigid1 && !rigid2 && only_r2.is_empty() {
                    let row = EffectRow::new(self.db, only_r1, Some(v1));
                    self.row_subst.insert(v2.id, row);
                    return Ok(());
                }
                if rigid2 && !rigid1 && only_r1.is_empty() {
                    let row = EffectRow::new(self.db, only_r2, Some(v2));
                    self.row_subst.insert(v1.id, row);
                    return Ok(());
                }

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
        // An effect with no candidate fails whatever later substitutions
        // decide, so it outranks ambiguity in any effect order.
        let mut ambiguous = false;
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
                    _ => ambiguous = true,
                }
            }
        }
        if ambiguous {
            return Err(SolveError::AmbiguousEffect {
                expected: r1,
                actual: r2,
            });
        }

        // Do not let early matches erase ambiguity in the reverse direction.
        for (a, b) in pairs {
            self.unify_types(a, b)?;
        }
        Ok(())
    }

    /// Compute the effects of `list2` that have no candidate in `list1`.
    ///
    /// The arguments of an effect matched with its single candidate are
    /// added to `pairs` rather than unified here: a substitution made for one
    /// effect must not decide the candidates of another, so the caller
    /// unifies them once every effect has been matched.
    pub(super) fn compute_effect_difference(
        &self,
        list1: &[crate::ast::Effect<'db>],
        list2: &[crate::ast::Effect<'db>],
        pairs: &mut Vec<(Type<'db>, Type<'db>)>,
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
                [e1] => pairs.extend(e1.args.iter().copied().zip(e2.args.iter().copied())),
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
        let mut pairs = Vec::new();
        let only_list1 = self.compute_effect_difference(list2, list1, &mut pairs)?;
        let only_list2 = self.compute_effect_difference(list1, list2, &mut pairs)?;
        self.unify_effect_args(pairs)?;
        Ok((only_list1, only_list2))
    }

    fn unify_effect_args(
        &mut self,
        pairs: Vec<(Type<'db>, Type<'db>)>,
    ) -> Result<(), SolveError<'db>> {
        for (a, b) in pairs {
            self.unify_types(a, b)?;
        }
        Ok(())
    }
}
