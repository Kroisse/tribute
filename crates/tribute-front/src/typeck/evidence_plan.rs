//! Evidence selection of calls and resumes.
//!
//! A callable receives evidence shaped by its own effect row: the handler of
//! each explicit ability instance is the top marker of that instance, and the
//! row tail's handlers lie beneath. A call keeps that shape for its callee by
//! counting, for each instance explicit in the caller's row, how many callee
//! positions take the caller's handler (`new-plans/type-inference.md`,
//! 호출의 evidence 선택):
//!
//! - none, and the callee's row is open: `mask`, so an operation arriving
//!   through the callee's tail reaches the handler beneath;
//! - one: the evidence is passed unchanged;
//! - more: `dup` for each extra position.
//!
//! A handle body sees the enclosing evidence with a new handler for each
//! handled instance. When the enclosing row names a handled instance
//! explicitly, that handler is first masked: the body row names the instance
//! once, so nothing in the body can reach the hidden handler, and leaving it
//! would stand between the new handler and the row tail's handlers.
//!
//! Positions are counted from the callee's declared explicit instances and
//! the explicit instances its row tail is instantiated with. Applying the
//! row substitution to the callee's whole row would merge an instance named
//! in both, losing the second position.
//!
//! A callee's row stays open only when its tail is the caller's own tail:
//! that is the only way an operation can pass the callee's explicit
//! instances and still reach a handler of the caller. A tail variable that
//! solving left unconstrained can be instantiated with the empty row.
//!
//! A callable whose row tail is a declared union of row tails receives the
//! evidence of each of those tails beside its own, in the order the union
//! declares them. A call of such a callee gives each tail the caller's
//! evidence after that tail's own selection, and the callee's calls whose
//! callee row ends in one of those tails select that tail's evidence. A
//! selected tail holds none of the handlers the callable itself names, so
//! each position that takes one pushes it.
//!
//! Checking records each call against the evidence scope it runs in and
//! computes the selections only after solving, when every row is known.

use rustc_hash::FxHashMap as HashMap;

use crate::ast::{Effect, EffectRow, EffectVar, LocalId, NodeId, RowUnion};

use super::EvidenceStep;

/// The evidence a piece of code runs with.
#[derive(Clone, Debug)]
enum EvidenceScope<'db> {
    /// A function or lambda body, with its signature row once it is known.
    Callable(Option<EffectRow<'db>>),
    /// A handle body: the enclosing scope's evidence with a new handler for
    /// each handled instance.
    HandleBody {
        parent: usize,
        handled: Option<EffectRow<'db>>,
    },
}

#[derive(Clone, Debug)]
enum EvidenceSite<'db> {
    /// A call whose callee has this (unsolved) row.
    Call {
        scope: usize,
        callee: EffectRow<'db>,
    },
    /// A resume made from `scope` into the computation of the handle body
    /// `body`.
    Resume { scope: usize, body: usize },
    /// The installation of the handle whose body is `body`.
    Handle { body: usize },
}

/// The evidence scope that code currently runs in, saved across a nested scope.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SavedEvidenceScope {
    current: Option<usize>,
    resume_from: Option<usize>,
}

/// Evidence scopes and call sites of one function body.
#[derive(Debug, Default)]
pub(crate) struct EvidenceTracker<'db> {
    scopes: Vec<EvidenceScope<'db>>,
    current: Option<usize>,
    /// The scope a `resume` passes evidence from: the handle body for a
    /// resume in an arm body, the lambda for a resume in a lambda in the arm.
    resume_from: Option<usize>,
    /// The handle body each continuation local resumes.
    continuations: HashMap<LocalId, usize>,
    sites: HashMap<NodeId, EvidenceSite<'db>>,
    /// The unions of every instantiated scheme, with their unsolved rows.
    unions: Vec<RowUnion<'db>>,
    /// The unions the checked function's own signature declares.
    own_unions: Vec<RowUnion<'db>>,
}

/// The evidence of a scope: its explicit instances, the row tail beneath
/// them, and the row tails that tail is a declared union of.
struct ScopeEvidence<'db> {
    explicit: Vec<Effect<'db>>,
    tail: Option<EffectVar>,
    tails: Vec<EffectVar>,
}

impl<'db> EvidenceTracker<'db> {
    fn save(&self) -> SavedEvidenceScope {
        SavedEvidenceScope {
            current: self.current,
            resume_from: self.resume_from,
        }
    }

    /// Restore the scope that was current before a nested scope.
    pub(crate) fn restore(&mut self, saved: SavedEvidenceScope) {
        self.current = saved.current;
        self.resume_from = saved.resume_from;
    }

    /// Enter a function or lambda body. The row may be supplied later with
    /// [`Self::set_callable_row`] when it is inferred from the body.
    pub(crate) fn enter_callable(
        &mut self,
        row: Option<EffectRow<'db>>,
    ) -> (usize, SavedEvidenceScope) {
        let saved = self.save();
        let scope = self.scopes.len();
        self.scopes.push(EvidenceScope::Callable(row));
        self.current = Some(scope);
        // A resume in a lambda passes the lambda's own evidence.
        if self.resume_from.is_some() {
            self.resume_from = Some(scope);
        }
        (scope, saved)
    }

    pub(crate) fn set_callable_row(&mut self, scope: usize, row: EffectRow<'db>) {
        if let Some(EvidenceScope::Callable(slot)) = self.scopes.get_mut(scope) {
            *slot = Some(row);
        }
    }

    /// Enter a handle body. Its handled instances may be supplied later with
    /// [`Self::set_handled`] when they are known only after the body.
    pub(crate) fn enter_handle_body(
        &mut self,
        handle: NodeId,
        handled: Option<EffectRow<'db>>,
    ) -> Option<(usize, SavedEvidenceScope)> {
        let parent = self.current?;
        let saved = self.save();
        let scope = self.scopes.len();
        self.scopes
            .push(EvidenceScope::HandleBody { parent, handled });
        self.current = Some(scope);
        self.sites
            .entry(handle)
            .or_insert(EvidenceSite::Handle { body: scope });
        Some((scope, saved))
    }

    pub(crate) fn set_handled(&mut self, scope: usize, row: EffectRow<'db>) {
        if let Some(EvidenceScope::HandleBody { handled, .. }) = self.scopes.get_mut(scope) {
            *handled = Some(row);
        }
    }

    /// Enter an `op` arm body of the handle whose body is `body`. The arm's
    /// own operations and calls keep the enclosing scope; its resumes pass the
    /// handle body's evidence.
    pub(crate) fn enter_op_arm(&mut self, body: usize) -> SavedEvidenceScope {
        let saved = self.save();
        self.resume_from = Some(body);
        saved
    }

    /// Bind the continuation local of an `op` arm to the handle body it resumes.
    pub(crate) fn bind_continuation(&mut self, local: LocalId, body: usize) {
        self.continuations.insert(local, body);
    }

    /// Record the unions of an instantiated scheme.
    pub(crate) fn record_unions(&mut self, unions: &[RowUnion<'db>]) {
        self.unions.extend_from_slice(unions);
    }

    /// Record the unions of the checked function's own signature.
    pub(crate) fn record_own_unions(&mut self, unions: &[RowUnion<'db>]) {
        self.own_unions.extend_from_slice(unions);
    }

    /// Record a call whose callee has the given row.
    pub(crate) fn record_call(&mut self, call: NodeId, callee: EffectRow<'db>) {
        if let Some(scope) = self.current {
            self.sites
                .entry(call)
                .or_insert(EvidenceSite::Call { scope, callee });
        }
    }

    /// Record a resume of the continuation bound to `local`.
    pub(crate) fn record_resume(&mut self, resume: NodeId, local: LocalId) {
        let Some(body) = self.continuations.get(&local).copied() else {
            return;
        };
        let scope = self.resume_from.unwrap_or(body);
        self.sites
            .entry(resume)
            .or_insert(EvidenceSite::Resume { scope, body });
    }

    /// Compute the non-identity selections once every row is solved.
    ///
    /// `resolve` applies the solution to a row, including the type arguments
    /// of its instances.
    pub(crate) fn plans(
        &self,
        db: &'db dyn salsa::Database,
        resolve: impl Fn(EffectRow<'db>) -> EffectRow<'db>,
    ) -> HashMap<NodeId, Vec<EvidenceStep<'db>>> {
        let mut explicit = HashMap::default();
        let mut plans = HashMap::default();
        for (node, site) in &self.sites {
            let plan = match site {
                EvidenceSite::Call { scope, callee } => {
                    let (explicit, tail) = self.explicit(db, *scope, &resolve, &mut explicit);
                    let caller = ScopeEvidence {
                        tails: self.declared_tails(db, tail, &resolve),
                        explicit,
                        tail,
                    };
                    let (positions, callee_tail) = callee_positions(db, *callee, &resolve);
                    let mut plan = caller.select(&positions, callee_tail);
                    if let Some(sources) = self.callee_tails(db, *callee) {
                        let plans = sources
                            .map(|source| {
                                let source = resolve(source);
                                let positions = unique(source.effects(db).iter().cloned());
                                caller.select(&positions, source.rest(db))
                            })
                            .collect();
                        plan.push(EvidenceStep::Tails(plans));
                    }
                    plan
                }
                EvidenceSite::Resume { scope, body } if scope != body => {
                    let (caller, tail) = self.explicit(db, *scope, &resolve, &mut explicit);
                    let (body, body_tail) = self.explicit(db, *body, &resolve, &mut explicit);
                    // The handle body takes each of its instances once.
                    call_plan(&caller, &body, opens_into(body_tail, tail))
                }
                EvidenceSite::Resume { .. } => Vec::new(),
                EvidenceSite::Handle { body } => {
                    let EvidenceScope::HandleBody { parent, handled } = &self.scopes[*body] else {
                        unreachable!("a handle site names a handle body scope");
                    };
                    let (parent, _) = self.explicit(db, *parent, &resolve, &mut explicit);
                    handled
                        .map(|row| resolve(row).effects(db).to_vec())
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|instance| parent.contains(instance))
                        .map(EvidenceStep::Mask)
                        .collect()
                }
            };
            if !plan.is_empty() {
                plans.insert(*node, plan);
            }
        }
        plans
    }

    /// The row tails that `tail` is a union of in the checked function's
    /// signature, in declaration order.
    fn declared_tails(
        &self,
        db: &'db dyn salsa::Database,
        tail: Option<EffectVar>,
        resolve: &impl Fn(EffectRow<'db>) -> EffectRow<'db>,
    ) -> Vec<EffectVar> {
        let Some(tail) = tail else {
            return Vec::new();
        };
        self.own_unions
            .iter()
            .filter(|union| resolve(union.result).rest(db) == Some(tail))
            .find_map(|union| {
                let tails: Vec<_> = open_sources(db, union)
                    .map(|source| resolve(source).rest(db).filter(|source| *source != tail))
                    .collect::<Option<_>>()?;
                (tails.len() > 1).then_some(tails)
            })
            .unwrap_or_default()
    }

    /// The rows a callee's row tail is a union of, when its scheme declares
    /// it as a union of row tails.
    fn callee_tails(
        &self,
        db: &'db dyn salsa::Database,
        callee: EffectRow<'db>,
    ) -> Option<impl Iterator<Item = EffectRow<'db>>> {
        let tail = callee.rest(db)?;
        self.unions
            .iter()
            .find(|union| {
                union.result.rest(db) == Some(tail) && open_sources(db, union).count() > 1
            })
            .map(|union| open_sources(db, union))
    }

    /// The explicit instances of a scope's evidence, and the row tail that
    /// lies beneath them.
    fn explicit(
        &self,
        db: &'db dyn salsa::Database,
        scope: usize,
        resolve: &impl Fn(EffectRow<'db>) -> EffectRow<'db>,
        memo: &mut HashMap<usize, (Vec<Effect<'db>>, Option<EffectVar>)>,
    ) -> (Vec<Effect<'db>>, Option<EffectVar>) {
        if let Some(found) = memo.get(&scope) {
            return found.clone();
        }
        let found = match &self.scopes[scope] {
            EvidenceScope::Callable(Some(row)) => {
                let row = resolve(*row);
                // The ambient `Io` has no handler marker to select.
                let selectable = row
                    .effects(db)
                    .iter()
                    .filter(|instance| !instance.ability_id.is_builtin_io(db))
                    .cloned();
                (unique(selectable), row.rest(db))
            }
            EvidenceScope::Callable(None) => (Vec::new(), None),
            EvidenceScope::HandleBody { parent, handled } => {
                let (parent, tail) = self.explicit(db, *parent, resolve, memo);
                let handled = handled
                    .map(|row| resolve(row).effects(db).to_vec())
                    .unwrap_or_default();
                (unique(handled.into_iter().chain(parent)), tail)
            }
        };
        memo.insert(scope, found.clone());
        found
    }
}

/// The callee positions that a caller's explicit instances may fill, and
/// the callee's row tail after instantiation.
fn callee_positions<'db>(
    db: &'db dyn salsa::Database,
    callee: EffectRow<'db>,
    resolve: &impl Fn(EffectRow<'db>) -> EffectRow<'db>,
) -> (Vec<Effect<'db>>, Option<EffectVar>) {
    // Resolve the declared instances and the tail separately so an instance
    // named by both keeps both positions. Within one part, an instance takes
    // one position however many times solving left it in the row.
    let declared = resolve(EffectRow::new(db, callee.effects(db).to_vec(), None));
    let mut positions = unique(declared.effects(db).iter().cloned());
    let tail = callee.rest(db).and_then(|tail| {
        let tail = resolve(EffectRow::open(db, tail));
        positions.extend(unique(tail.effects(db).iter().cloned()));
        tail.rest(db)
    });
    (positions, tail)
}

/// The sources of a union that its declaration leaves open: one row tail
/// each. Callers and the callee number the tails by this order.
fn open_sources<'db>(
    db: &'db dyn salsa::Database,
    union: &RowUnion<'db>,
) -> impl Iterator<Item = EffectRow<'db>> {
    union
        .sources
        .iter()
        .copied()
        .filter(move |source| source.rest(db).is_some())
}

impl<'db> ScopeEvidence<'db> {
    /// Select the evidence of a row with the explicit `positions` and the
    /// row tail `target` from this scope's evidence.
    fn select(
        &self,
        positions: &[Effect<'db>],
        target: Option<EffectVar>,
    ) -> Vec<EvidenceStep<'db>> {
        let index = target.and_then(|target| self.tails.iter().position(|tail| *tail == target));
        let Some(index) = index else {
            return call_plan(&self.explicit, positions, opens_into(target, self.tail));
        };
        // The selected tail holds none of this scope's explicit handlers.
        let mut plan = vec![EvidenceStep::Select(index as u32)];
        for instance in &self.explicit {
            let count = positions
                .iter()
                .filter(|position| *position == instance)
                .count();
            if count > 0 {
                plan.push(EvidenceStep::Push(instance.clone()));
                plan.extend((1..count).map(|_| EvidenceStep::Dup(instance.clone())));
            }
        }
        plan
    }
}

/// Whether operations through a callee's row tail reach the caller's tail.
fn opens_into(callee_tail: Option<EffectVar>, caller_tail: Option<EffectVar>) -> bool {
    callee_tail.is_some() && callee_tail == caller_tail
}

/// Select the evidence for a callee from the caller's explicit instances.
fn call_plan<'db>(
    caller: &[Effect<'db>],
    positions: &[Effect<'db>],
    open: bool,
) -> Vec<EvidenceStep<'db>> {
    let mut plan = Vec::new();
    for instance in caller {
        match positions
            .iter()
            .filter(|position| *position == instance)
            .count()
        {
            0 if open => plan.push(EvidenceStep::Mask(instance.clone())),
            0 | 1 => {}
            count => plan.extend((1..count).map(|_| EvidenceStep::Dup(instance.clone()))),
        }
    }
    plan
}

fn unique<'db>(effects: impl IntoIterator<Item = Effect<'db>>) -> Vec<Effect<'db>> {
    let mut unique = Vec::new();
    for effect in effects {
        if !unique.contains(&effect) {
            unique.push(effect);
        }
    }
    unique
}
