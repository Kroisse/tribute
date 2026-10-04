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
//! Checking records each call against the evidence scope it runs in and
//! computes the selections only after solving, when every row is known.

use hashbrown::HashMap;

use crate::ast::{Effect, EffectRow, EffectVar, LocalId, NodeId};

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
        let mut explicit = HashMap::new();
        let mut plans = HashMap::new();
        for (node, site) in &self.sites {
            let plan = match site {
                EvidenceSite::Call { scope, callee } => {
                    let (caller, tail) = self.explicit(db, *scope, &resolve, &mut explicit);
                    let (positions, callee_tail) = callee_positions(db, *callee, &resolve);
                    call_plan(&caller, &positions, opens_into(callee_tail, tail))
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
