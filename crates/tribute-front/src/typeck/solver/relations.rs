//! Deferred common-result relations.

use super::{Constraint, ConstraintOrigin, LocatedSolveError, Type, TypeKind, TypeSolver};

impl<'db> TypeSolver<'db> {
    /// Nested cases in a resumptive arm can refer back to the handle answer.
    /// Resolve a closed strongly connected group from its independent sources.
    pub(super) fn resolve_join_cycle(&mut self) -> Result<bool, LocatedSolveError<'db>> {
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

    pub(super) fn solve_join(
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
}
