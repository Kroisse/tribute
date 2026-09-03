//! Analysis framework for TrunkIR passes.
//!
//! [`AnalysisCache`] provides lazy, cached analyses with explicit
//! invalidation semantics. Types implementing the [`Analysis`] trait are
//! computed on demand and cached keyed by `(TypeId, OpRef)`; passes that
//! mutate the IR are expected to call [`AnalysisCache::invalidate`] to
//! keep downstream consumers correct.
//!
//! A lookup either returns a complete cached result or an [`AnalysisError`].
//! Analysis errors describe invalid input IR or an unsupported analysis
//! boundary, not ordinary source-language diagnostics. A failed computation
//! is never inserted: a later lookup retries from scratch after the IR is
//! repaired or changed, while cached results for other analysis types and
//! targets remain available. This also means the type-erased cache can never
//! expose a partially constructed result. Unchanged invalid IR is expected to
//! fail again; cache type mismatches and caller-contract violations remain
//! programming errors and panic.
//!
//! Design inspired by MLIR's `AnalysisManager`. See issue #679 for
//! context and the follow-up roadmap (#680 hybrid inliner, #676
//! canonicalize).
//!
//! # Scope: pipeline-phase, injected into passes
//!
//! An [`AnalysisCache`] is owned by the **pipeline orchestrator** for
//! the duration of one pipeline phase and **injected** into each pass
//! that needs it. The cache is short-lived — dropped when the phase
//! returns — so cached [`OpRef`] keys never outlive the [`IrContext`]
//! they refer to, and the "one cache = one context" invariant holds by
//! construction rather than by a runtime guard.
//!
//! The [`AnalysisCache::scope`] helper bundles this pattern:
//!
//! ```ignore
//! fn run_cleanup_passes(ctx: &mut IrContext, m: Module) {
//!     AnalysisCache::scope(ctx, |ctx, analyses| {
//!         inline_functions(ctx, m, InlineConfig::default(), analyses);
//!         // canonicalize(ctx, m, analyses); — future pass sharing `analyses`
//!     });
//! }
//! ```
//!
//! [`AnalysisCache::new`] is also available for tests or ad-hoc use,
//! but orchestration code should prefer `scope` to make the phase
//! boundary explicit.
//!
//! # Usage
//!
//! ```ignore
//! use trunk_ir::analysis::AnalysisCache;
//! use trunk_ir::transforms::CallGraph;
//!
//! let mut analyses = AnalysisCache::new();
//! let graph = analyses.get::<CallGraph>(ctx, module.op())?;
//! // `graph: Arc<CallGraph>` — safe to hold while `ctx` is mutated.
//! do_some_mutation(ctx);
//! analyses.invalidate::<CallGraph>(module.op());
//! ```
//!
//! # Thread-safety
//!
//! `AnalysisCache` is not shared across threads; analyses are stored as
//! `Arc<dyn Any + Send + Sync>` for future flexibility, but the cache
//! itself is single-threaded.

use std::any::{Any, TypeId, type_name};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::context::IrContext;
use crate::refs::OpRef;

/// Internal IR-analysis failure returned by [`AnalysisCache::get`].
///
/// The wrapper records the concrete analysis type and requested target in
/// addition to preserving the analysis-specific error as its source.
#[derive(Debug)]
pub struct AnalysisError {
    analysis_type: &'static str,
    target: OpRef,
    source: Box<dyn Error + Send + Sync>,
    cycle: Option<AnalysisCycle>,
}

impl AnalysisError {
    /// Attach an analysis-specific source error to its exact analysis type
    /// and target operation.
    pub fn new<A>(target: OpRef, source: impl Error + Send + Sync + 'static) -> Self
    where
        A: Analysis,
    {
        Self {
            analysis_type: type_name::<A>(),
            target,
            source: Box::new(source),
            cycle: None,
        }
    }

    fn construction_cycle<A: Analysis>(target: OpRef, queries: Vec<AnalysisQuery>) -> Self {
        let cycle = AnalysisCycle { queries };
        Self {
            analysis_type: type_name::<A>(),
            target,
            source: Box::new(cycle.clone()),
            cycle: Some(cycle),
        }
    }

    /// Concrete Rust type of the analysis whose computation failed.
    pub fn analysis_type(&self) -> &'static str {
        self.analysis_type
    }

    /// Operation for which the failed analysis was requested.
    pub fn target(&self) -> OpRef {
        self.target
    }

    /// Return the construction cycle, if this failure was caused by one.
    pub fn cycle(&self) -> Option<&AnalysisCycle> {
        self.cycle.as_ref()
    }
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "analysis {} failed for {}: {}",
            self.analysis_type, self.target, self.source
        )
    }
}

impl Error for AnalysisError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}

/// One analysis query participating in a construction cycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalysisQuery {
    analysis_type: &'static str,
    target: OpRef,
}

impl AnalysisQuery {
    /// Concrete Rust type of the requested analysis.
    pub fn analysis_type(&self) -> &'static str {
        self.analysis_type
    }

    /// Operation requested by this query.
    pub fn target(&self) -> OpRef {
        self.target
    }
}

/// Structured description of an analysis-construction cycle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnalysisCycle {
    queries: Vec<AnalysisQuery>,
}

impl AnalysisCycle {
    /// The closed query path. The first and last entries are the same query.
    pub fn queries(&self) -> &[AnalysisQuery] {
        &self.queries
    }
}

impl fmt::Display for AnalysisCycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "analysis construction cycle: ")?;
        for (index, query) in self.queries.iter().enumerate() {
            if index != 0 {
                write!(f, " -> ")?;
            }
            write!(f, "{} for {}", query.analysis_type, query.target)?;
        }
        Ok(())
    }
}

impl Error for AnalysisCycle {}

/// An analysis computable from an IR context plus a target operation
/// (typically a `core.module` op).
///
/// Analyses should be pure functions of the IR state: computing twice on
/// an unchanged context must produce equivalent results. `compute` is a
/// typed static factory: the analysis type is both the query key and the
/// complete concrete result stored by [`AnalysisCache`].
pub trait Analysis: Any + Send + Sync {
    /// Compute this analysis for `target` through `ctx`.
    ///
    /// Return an error for invalid input IR or an unsupported analysis
    /// boundary. Construct failures with [`AnalysisError::new`] using this
    /// analysis type so dependent queries can preserve their original
    /// analysis and target context. Violated caller contracts and impossible
    /// implementation invariants must panic rather than be represented as
    /// analysis failures.
    fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError>
    where
        Self: Sized;
}

/// Read-only IR access and dependent-analysis lookup for one computation.
///
/// A context is created only while one [`Analysis::compute`] call is active.
/// Dependent lookups use its owning cache and are recorded atomically if the
/// enclosing computation succeeds.
pub struct AnalysisContext<'a> {
    ir: &'a IrContext,
    cache: &'a mut AnalysisCache,
    dependencies: HashSet<AnalysisKey>,
}

impl<'a> AnalysisContext<'a> {
    /// The IR context being analysed. It cannot be mutated through this API.
    pub fn ir(&self) -> &IrContext {
        self.ir
    }

    /// Get a typed prerequisite through the same cache.
    pub fn get<A: Analysis>(&mut self, target: OpRef) -> Result<Arc<A>, AnalysisError> {
        let analysis = self.cache.get::<A>(self.ir, target)?;
        self.dependencies.insert(AnalysisKey::of::<A>(target));
        Ok(analysis)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct AnalysisKey {
    analysis_type: TypeId,
    target: OpRef,
}

impl AnalysisKey {
    fn of<A: Analysis>(target: OpRef) -> Self {
        Self {
            analysis_type: TypeId::of::<A>(),
            target,
        }
    }
}

#[derive(Clone, Copy)]
struct InProgressAnalysis {
    key: AnalysisKey,
    analysis_type: &'static str,
}

/// Lazy, typed cache of analyses keyed by `(TypeId, OpRef)`.
///
/// See the [module docs](self) for the pipeline-scoped ownership
/// model and the single-context invariant.
#[derive(Default)]
pub struct AnalysisCache {
    cache: HashMap<AnalysisKey, Arc<dyn Any + Send + Sync>>,
    /// Computed analysis -> its direct prerequisites.
    dependencies: HashMap<AnalysisKey, HashSet<AnalysisKey>>,
    /// Prerequisite -> analyses that directly depend on it.
    dependents: HashMap<AnalysisKey, HashSet<AnalysisKey>>,
    in_progress: Vec<InProgressAnalysis>,
}

impl AnalysisCache {
    /// Create an empty cache. Prefer [`Self::scope`] in pipeline code.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run `f` with a fresh cache scoped to this pipeline phase.
    ///
    /// The cache is constructed alongside `ctx`, passed into `f`, and
    /// dropped when `f` returns. This encodes the "one cache per
    /// pipeline phase, bound to one `IrContext`" convention at the
    /// call site so passes in the same phase can share cached
    /// analyses without the orchestrator having to juggle lifetimes by
    /// hand.
    ///
    /// ```ignore
    /// AnalysisCache::scope(ctx, |ctx, analyses| {
    ///     inline_functions(ctx, m, InlineConfig::default(), analyses);
    ///     // more passes sharing `analyses`…
    /// });
    /// ```
    pub fn scope<R>(
        ctx: &mut IrContext,
        f: impl FnOnce(&mut IrContext, &mut AnalysisCache) -> R,
    ) -> R {
        let mut analyses = AnalysisCache::new();
        f(ctx, &mut analyses)
    }

    /// Compute (or return cached) analysis `A` for `target`.
    ///
    /// Returns an `Arc` so callers may hold the result across IR
    /// mutations without keeping the cache borrowed. On an internal
    /// IR-analysis failure, returns [`AnalysisError`] without publishing a
    /// cache entry; retry after repairing or changing the IR recomputes it.
    pub fn get<A: Analysis>(
        &mut self,
        ctx: &IrContext,
        target: OpRef,
    ) -> Result<Arc<A>, AnalysisError> {
        let key = AnalysisKey::of::<A>(target);
        if let Some(entry) = self.cache.get(&key) {
            return Ok(Arc::clone(entry)
                .downcast::<A>()
                .expect("analysis cache type mismatch"));
        }

        if let Some(cycle_start) = self.in_progress.iter().position(|entry| entry.key == key) {
            let mut queries = self.in_progress[cycle_start..]
                .iter()
                .map(|entry| AnalysisQuery {
                    analysis_type: entry.analysis_type,
                    target: entry.key.target,
                })
                .collect::<Vec<_>>();
            queries.push(AnalysisQuery {
                analysis_type: type_name::<A>(),
                target,
            });
            return Err(AnalysisError::construction_cycle::<A>(target, queries));
        }

        self.in_progress.push(InProgressAnalysis {
            key,
            analysis_type: type_name::<A>(),
        });
        let (result, dependencies) = {
            let mut computation = AnalysisContext {
                ir: ctx,
                cache: self,
                dependencies: HashSet::new(),
            };
            let result = A::compute(&mut computation, target);
            (result, computation.dependencies)
        };
        let completed = self
            .in_progress
            .pop()
            .expect("analysis construction stack unexpectedly empty");
        assert_eq!(completed.key, key, "analysis construction stack corrupted");

        // Publish only a complete result and its complete dependency set. A
        // failure leaves neither a cache entry nor dependency metadata behind.
        let analysis = Arc::new(result?);
        let entry: Arc<dyn Any + Send + Sync> = analysis.clone();
        self.replace_dependencies(key, dependencies);
        self.cache.insert(key, entry);
        Ok(analysis)
    }

    /// Return the cached analysis `A` for `target` without computing it.
    pub fn get_cached<A: Analysis>(&self, target: OpRef) -> Option<Arc<A>> {
        let key = AnalysisKey::of::<A>(target);
        self.cache.get(&key).map(|v| {
            Arc::clone(v)
                .downcast::<A>()
                .expect("analysis cache type mismatch")
        })
    }

    /// Invalidate the cached analysis `A` for `target`, if present.
    ///
    /// Every analysis that transitively depends on it is also invalidated,
    /// including analyses of other types and other targets. Prerequisites of
    /// the invalidated analyses are kept.
    pub fn invalidate<A: Analysis>(&mut self, target: OpRef) {
        self.invalidate_keys([AnalysisKey::of::<A>(target)]);
    }

    /// Invalidate every cached analysis for `target`, and every analysis that
    /// transitively depends on them. Dependents attached to other targets are
    /// invalidated too; prerequisites are kept.
    pub fn invalidate_all(&mut self, target: OpRef) {
        let keys = self
            .cache
            .keys()
            .filter(|key| key.target == target)
            .copied()
            .collect::<Vec<_>>();
        self.invalidate_keys(keys);
    }

    /// Drop every cached analysis.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.dependencies.clear();
        self.dependents.clear();
        self.in_progress.clear();
    }

    /// Number of cached analyses (useful for diagnostics/tests).
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    fn replace_dependencies(&mut self, key: AnalysisKey, dependencies: HashSet<AnalysisKey>) {
        self.remove_dependencies(key);
        if dependencies.is_empty() {
            return;
        }
        for prerequisite in &dependencies {
            self.dependents
                .entry(*prerequisite)
                .or_default()
                .insert(key);
        }
        self.dependencies.insert(key, dependencies);
    }

    fn remove_dependencies(&mut self, key: AnalysisKey) {
        let Some(prerequisites) = self.dependencies.remove(&key) else {
            return;
        };
        for prerequisite in prerequisites {
            let remove_entry = self
                .dependents
                .get_mut(&prerequisite)
                .is_some_and(|dependents| {
                    dependents.remove(&key);
                    dependents.is_empty()
                });
            if remove_entry {
                self.dependents.remove(&prerequisite);
            }
        }
    }

    fn invalidate_keys(&mut self, roots: impl IntoIterator<Item = AnalysisKey>) {
        let mut invalidated = HashSet::new();
        let mut pending = roots.into_iter().collect::<Vec<_>>();
        while let Some(key) = pending.pop() {
            if !invalidated.insert(key) {
                continue;
            }
            if let Some(dependents) = self.dependents.get(&key) {
                pending.extend(dependents.iter().copied());
            }
        }

        for key in &invalidated {
            self.cache.remove(key);
            self.remove_dependencies(*key);
        }
        for key in invalidated {
            self.dependents.remove(&key);
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    fn test_ctx() -> (IrContext, OpRef) {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, "core.module @test {}");
        let op = module.op();
        (ctx, op)
    }

    #[derive(Debug)]
    struct DummyAnalysis {
        target: OpRef,
    }

    impl Analysis for DummyAnalysis {
        fn compute(_ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            Ok(Self { target })
        }
    }

    #[derive(Debug)]
    struct OtherAnalysis;

    impl Analysis for OtherAnalysis {
        fn compute(_ctx: &mut AnalysisContext<'_>, _target: OpRef) -> Result<Self, AnalysisError> {
            Ok(Self)
        }
    }

    #[derive(Debug, derive_more::Display, derive_more::Error)]
    #[display("test analysis failed")]
    struct TestAnalysisError;

    static COUNTED_COMPUTES: AtomicUsize = AtomicUsize::new(0);
    static LEAF_COMPUTES: AtomicUsize = AtomicUsize::new(0);
    static MIDDLE_COMPUTES: AtomicUsize = AtomicUsize::new(0);
    static ROOT_COMPUTES: AtomicUsize = AtomicUsize::new(0);
    static COUNTER_LOCK: Mutex<()> = Mutex::new(());

    fn lock_counters() -> std::sync::MutexGuard<'static, ()> {
        COUNTER_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[derive(Debug)]
    struct CountedAnalysis;

    impl Analysis for CountedAnalysis {
        fn compute(_ctx: &mut AnalysisContext<'_>, _target: OpRef) -> Result<Self, AnalysisError> {
            COUNTED_COMPUTES.fetch_add(1, Ordering::SeqCst);
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct LeafAnalysis;

    impl Analysis for LeafAnalysis {
        fn compute(_ctx: &mut AnalysisContext<'_>, _target: OpRef) -> Result<Self, AnalysisError> {
            LEAF_COMPUTES.fetch_add(1, Ordering::SeqCst);
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct MiddleAnalysis;

    impl Analysis for MiddleAnalysis {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            MIDDLE_COMPUTES.fetch_add(1, Ordering::SeqCst);
            let _ = ctx.get::<LeafAnalysis>(target)?;
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct RootAnalysis;

    impl Analysis for RootAnalysis {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            ROOT_COMPUTES.fetch_add(1, Ordering::SeqCst);
            let _ = ctx.get::<MiddleAnalysis>(target)?;
            let _ = ctx.get::<LeafAnalysis>(target)?;
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct DirectCycleAnalysis;

    impl Analysis for DirectCycleAnalysis {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let _ = ctx.get::<Self>(target)?;
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct IndirectCycleFirst;

    #[derive(Debug)]
    struct IndirectCycleSecond;

    impl Analysis for IndirectCycleFirst {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let _ = ctx.get::<IndirectCycleSecond>(target)?;
            Ok(Self)
        }
    }

    impl Analysis for IndirectCycleSecond {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let _ = ctx.get::<IndirectCycleFirst>(target)?;
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct CrossTargetPrerequisite;

    impl Analysis for CrossTargetPrerequisite {
        fn compute(_ctx: &mut AnalysisContext<'_>, _target: OpRef) -> Result<Self, AnalysisError> {
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct CrossTargetDependent;

    impl Analysis for CrossTargetDependent {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let body = ctx.ir().op(target).regions[0];
            let block = ctx.ir().region(body).blocks[0];
            let prerequisite_target = ctx.ir().block(block).ops[0];
            let _ = ctx.get::<CrossTargetPrerequisite>(prerequisite_target)?;
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct ChoicePrerequisiteFirst;

    impl Analysis for ChoicePrerequisiteFirst {
        fn compute(_ctx: &mut AnalysisContext<'_>, _target: OpRef) -> Result<Self, AnalysisError> {
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct ChoicePrerequisiteSecond;

    impl Analysis for ChoicePrerequisiteSecond {
        fn compute(_ctx: &mut AnalysisContext<'_>, _target: OpRef) -> Result<Self, AnalysisError> {
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct ChoiceDependent;

    impl Analysis for ChoiceDependent {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let body = ctx.ir().op(target).regions[0];
            let block = ctx.ir().region(body).blocks[0];
            let first_child = ctx.ir().block(block).ops[0];
            if ctx.ir().op(first_child).attributes.get_symbol("sym_name")
                == Some(crate::symbol::Symbol::new("first"))
            {
                let _ = ctx.get::<ChoicePrerequisiteFirst>(first_child)?;
            } else {
                let _ = ctx.get::<ChoicePrerequisiteSecond>(first_child)?;
            }
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct FailingDependent;

    impl Analysis for FailingDependent {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let _ = ctx.get::<LeafAnalysis>(target)?;
            Err(AnalysisError::new::<Self>(target, TestAnalysisError))
        }
    }

    #[derive(Debug)]
    struct NonEmptyModuleAnalysis;

    impl Analysis for NonEmptyModuleAnalysis {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let module = crate::rewrite::Module::new(ctx.ir(), target)
                .expect("test only requests this analysis for core.module targets");
            module
                .body(ctx.ir())
                .filter(|&body| !ctx.ir().region(body).blocks.is_empty())
                .map(|_| Self)
                .ok_or_else(|| AnalysisError::new::<Self>(target, TestAnalysisError))
        }
    }

    #[test]
    fn get_returns_same_arc_on_cache_hit() {
        let _counters = lock_counters();
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        COUNTED_COMPUTES.store(0, Ordering::SeqCst);

        let a1 = analyses.get::<CountedAnalysis>(&ctx, op).unwrap();
        let a2 = analyses.get::<CountedAnalysis>(&ctx, op).unwrap();

        // Cache hit: both calls return the very same `Arc` and compute runs
        // exactly once.
        assert!(Arc::ptr_eq(&a1, &a2));
        assert_eq!(COUNTED_COMPUTES.load(Ordering::SeqCst), 1);
        assert_eq!(analyses.len(), 1);
    }

    #[test]
    fn invalidate_forces_recompute() {
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();

        let a1 = analyses.get::<DummyAnalysis>(&ctx, op).unwrap();
        analyses.invalidate::<DummyAnalysis>(op);
        let a2 = analyses.get::<DummyAnalysis>(&ctx, op).unwrap();

        // After invalidation the cached entry is dropped, so the next
        // `get` rebuilds the analysis — a distinct `Arc` results.
        assert!(!Arc::ptr_eq(&a1, &a2));
        assert_eq!(a1.target, a2.target);
    }

    #[test]
    fn get_cached_returns_none_before_compute() {
        let (_ctx, op) = test_ctx();
        let analyses = AnalysisCache::new();
        assert!(analyses.get_cached::<DummyAnalysis>(op).is_none());
    }

    #[test]
    fn get_cached_returns_some_after_compute() {
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        let _ = analyses.get::<DummyAnalysis>(&ctx, op).unwrap();
        assert!(analyses.get_cached::<DummyAnalysis>(op).is_some());
    }

    #[test]
    fn invalidate_all_clears_all_analyses_for_target() {
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        let _ = analyses.get::<DummyAnalysis>(&ctx, op).unwrap();
        let _ = analyses.get::<OtherAnalysis>(&ctx, op).unwrap();
        assert_eq!(analyses.len(), 2);

        analyses.invalidate_all(op);
        assert!(analyses.is_empty());
    }

    #[test]
    fn different_analyses_cached_independently() {
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        let _ = analyses.get::<DummyAnalysis>(&ctx, op).unwrap();
        let _ = analyses.get::<OtherAnalysis>(&ctx, op).unwrap();
        assert_eq!(analyses.len(), 2);

        analyses.invalidate::<DummyAnalysis>(op);
        assert!(analyses.get_cached::<DummyAnalysis>(op).is_none());
        assert!(analyses.get_cached::<OtherAnalysis>(op).is_some());
    }

    #[test]
    fn clear_drops_every_entry() {
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        let _ = analyses.get::<DummyAnalysis>(&ctx, op).unwrap();
        let _ = analyses.get::<OtherAnalysis>(&ctx, op).unwrap();
        analyses.clear();
        assert!(analyses.is_empty());
    }

    #[test]
    fn scope_provides_ctx_and_cache_together() {
        let (mut ctx, op) = test_ctx();
        let len = AnalysisCache::scope(&mut ctx, |_ctx, analyses| {
            let _ = analyses.get::<DummyAnalysis>(_ctx, op).unwrap();
            analyses.len()
        });
        assert_eq!(len, 1);
    }

    #[test]
    fn failed_computation_is_not_cached_and_can_retry() {
        let mut ctx = IrContext::new();
        let op = crate::parser::parse_test_module(&mut ctx, "core.module @empty {}").op();
        let mut analyses = AnalysisCache::new();

        let error = analyses
            .get::<NonEmptyModuleAnalysis>(&ctx, op)
            .unwrap_err();
        assert_eq!(error.analysis_type(), type_name::<NonEmptyModuleAnalysis>());
        assert_eq!(error.target(), op);
        assert_eq!(error.source().unwrap().to_string(), "test analysis failed");
        assert!(analyses.get_cached::<NonEmptyModuleAnalysis>(op).is_none());
        assert!(analyses.is_empty());

        // Repair the analysis input before retrying. An unchanged malformed
        // module would fail again rather than becoming valid by retry alone.
        let location = ctx.op(op).location;
        let block = ctx.create_block(crate::context::BlockData {
            location,
            args: vec![],
            ops: smallvec::SmallVec::new(),
            parent_region: None,
        });
        let body = ctx.create_region(crate::context::RegionData {
            location,
            blocks: smallvec::smallvec![block],
            parent_op: Some(op),
        });
        ctx.op_mut(op).regions.push(body);

        let first_success = analyses.get::<NonEmptyModuleAnalysis>(&ctx, op).unwrap();
        let second_success = analyses.get::<NonEmptyModuleAnalysis>(&ctx, op).unwrap();
        assert!(Arc::ptr_eq(&first_success, &second_success));
        assert_eq!(analyses.len(), 1);
    }

    #[test]
    fn failure_isolated_from_other_targets_and_analysis_types() {
        let input = r#"core.module @outer {
  core.module @empty {
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input).op();
        let body = ctx.op(module).regions[0];
        let block = ctx.region(body).blocks[0];
        let empty_module = ctx.block(block).ops[0];
        let mut analyses = AnalysisCache::new();

        let _ = analyses.get::<DummyAnalysis>(&ctx, module).unwrap();
        let module_analysis = analyses
            .get::<NonEmptyModuleAnalysis>(&ctx, module)
            .unwrap();
        assert!(
            analyses
                .get::<NonEmptyModuleAnalysis>(&ctx, empty_module)
                .is_err()
        );

        assert!(
            analyses
                .get_cached::<NonEmptyModuleAnalysis>(empty_module)
                .is_none()
        );
        assert!(analyses.get_cached::<DummyAnalysis>(module).is_some());
        assert!(Arc::ptr_eq(
            &module_analysis,
            &analyses
                .get_cached::<NonEmptyModuleAnalysis>(module)
                .unwrap()
        ));
        assert_eq!(analyses.len(), 2);
    }

    #[test]
    fn dependent_chain_and_diamond_compute_each_key_once() {
        let _counters = lock_counters();
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        LEAF_COMPUTES.store(0, Ordering::SeqCst);
        MIDDLE_COMPUTES.store(0, Ordering::SeqCst);
        ROOT_COMPUTES.store(0, Ordering::SeqCst);

        let _ = analyses.get::<RootAnalysis>(&ctx, op).unwrap();
        let _ = analyses.get::<RootAnalysis>(&ctx, op).unwrap();

        assert_eq!(LEAF_COMPUTES.load(Ordering::SeqCst), 1);
        assert_eq!(MIDDLE_COMPUTES.load(Ordering::SeqCst), 1);
        assert_eq!(ROOT_COMPUTES.load(Ordering::SeqCst), 1);
        assert_eq!(analyses.len(), 3);
    }

    #[test]
    fn cached_prerequisite_lookup_records_direct_dependency() {
        let _counters = lock_counters();
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();

        let _ = analyses.get::<LeafAnalysis>(&ctx, op).unwrap();
        let _ = analyses.get::<MiddleAnalysis>(&ctx, op).unwrap();

        analyses.invalidate::<LeafAnalysis>(op);
        assert!(analyses.get_cached::<LeafAnalysis>(op).is_none());
        assert!(analyses.get_cached::<MiddleAnalysis>(op).is_none());
    }

    #[test]
    fn direct_and_indirect_cycles_are_structured_and_publish_nothing() {
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();

        let direct = analyses.get::<DirectCycleAnalysis>(&ctx, op).unwrap_err();
        let direct_queries = direct.cycle().unwrap().queries();
        assert_eq!(direct_queries.len(), 2);
        assert_eq!(
            direct_queries[0].analysis_type(),
            type_name::<DirectCycleAnalysis>()
        );
        assert_eq!(direct_queries[0].target(), op);
        assert_eq!(direct_queries[1], direct_queries[0]);
        assert!(analyses.is_empty());
        assert!(analyses.dependencies.is_empty());
        assert!(analyses.dependents.is_empty());

        let indirect = analyses.get::<IndirectCycleFirst>(&ctx, op).unwrap_err();
        let indirect_queries = indirect.cycle().unwrap().queries();
        assert_eq!(indirect_queries.len(), 3);
        assert_eq!(
            indirect_queries
                .iter()
                .map(AnalysisQuery::analysis_type)
                .collect::<Vec<_>>(),
            vec![
                type_name::<IndirectCycleFirst>(),
                type_name::<IndirectCycleSecond>(),
                type_name::<IndirectCycleFirst>(),
            ]
        );
        assert!(analyses.is_empty());
        assert!(analyses.dependencies.is_empty());
        assert!(analyses.dependents.is_empty());
    }

    #[test]
    fn prerequisite_invalidation_cascades_but_dependent_invalidation_preserves_prerequisites() {
        let _counters = lock_counters();
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        LEAF_COMPUTES.store(0, Ordering::SeqCst);
        MIDDLE_COMPUTES.store(0, Ordering::SeqCst);
        ROOT_COMPUTES.store(0, Ordering::SeqCst);
        let _ = analyses.get::<RootAnalysis>(&ctx, op).unwrap();

        analyses.invalidate::<RootAnalysis>(op);
        assert!(analyses.get_cached::<RootAnalysis>(op).is_none());
        assert!(analyses.get_cached::<MiddleAnalysis>(op).is_some());
        assert!(analyses.get_cached::<LeafAnalysis>(op).is_some());
        let _ = analyses.get::<RootAnalysis>(&ctx, op).unwrap();
        assert_eq!(ROOT_COMPUTES.load(Ordering::SeqCst), 2);
        assert_eq!(MIDDLE_COMPUTES.load(Ordering::SeqCst), 1);
        assert_eq!(LEAF_COMPUTES.load(Ordering::SeqCst), 1);

        analyses.invalidate::<LeafAnalysis>(op);
        assert!(analyses.is_empty());
        assert!(analyses.dependencies.is_empty());
        assert!(analyses.dependents.is_empty());
    }

    #[test]
    fn recomputation_replaces_obsolete_dependency_edges() {
        let mut ctx = IrContext::new();
        let outer = crate::parser::parse_test_module(
            &mut ctx,
            "core.module @outer { core.module @first {} core.module @second {} }",
        )
        .op();
        let body = ctx.op(outer).regions[0];
        let block = ctx.region(body).blocks[0];
        let first = ctx.block(block).ops[0];
        let second = ctx.block(block).ops[1];
        let mut analyses = AnalysisCache::new();

        let _ = analyses.get::<ChoiceDependent>(&ctx, outer).unwrap();
        assert!(
            analyses
                .get_cached::<ChoicePrerequisiteFirst>(first)
                .is_some()
        );
        ctx.block_mut(block).ops.swap(0, 1);
        analyses.invalidate::<ChoiceDependent>(outer);
        let _ = analyses.get::<ChoiceDependent>(&ctx, outer).unwrap();

        assert!(
            analyses
                .get_cached::<ChoicePrerequisiteSecond>(second)
                .is_some()
        );
        analyses.invalidate::<ChoicePrerequisiteFirst>(first);
        assert!(analyses.get_cached::<ChoiceDependent>(outer).is_some());
        analyses.invalidate::<ChoicePrerequisiteSecond>(second);
        assert!(analyses.get_cached::<ChoiceDependent>(outer).is_none());
    }

    #[test]
    fn failed_dependent_leaves_no_cache_entry_or_dependency_edge() {
        let _counters = lock_counters();
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();

        assert!(analyses.get::<FailingDependent>(&ctx, op).is_err());
        assert!(analyses.get_cached::<FailingDependent>(op).is_none());
        assert!(analyses.get_cached::<LeafAnalysis>(op).is_some());
        assert!(analyses.dependencies.is_empty());
        assert!(analyses.dependents.is_empty());

        analyses.invalidate::<LeafAnalysis>(op);
        assert!(analyses.is_empty());
    }

    #[test]
    fn cross_target_invalidate_all_cascades_but_preserves_unrelated_entries() {
        let mut ctx = IrContext::new();
        let outer = crate::parser::parse_test_module(
            &mut ctx,
            "core.module @outer { core.module @inner {} core.module @unrelated {} }",
        )
        .op();
        let body = ctx.op(outer).regions[0];
        let block = ctx.region(body).blocks[0];
        let inner = ctx.block(block).ops[0];
        let unrelated = ctx.block(block).ops[1];
        let mut analyses = AnalysisCache::new();

        let _ = analyses
            .get::<CrossTargetPrerequisite>(&ctx, inner)
            .unwrap();
        let _ = analyses.get::<CrossTargetDependent>(&ctx, outer).unwrap();
        let _ = analyses.get::<OtherAnalysis>(&ctx, inner).unwrap();
        let _ = analyses.get::<DummyAnalysis>(&ctx, outer).unwrap();
        let _ = analyses.get::<DummyAnalysis>(&ctx, unrelated).unwrap();
        analyses.invalidate_all(inner);

        assert!(
            analyses
                .get_cached::<CrossTargetPrerequisite>(inner)
                .is_none()
        );
        assert!(analyses.get_cached::<CrossTargetDependent>(outer).is_none());
        assert!(analyses.get_cached::<OtherAnalysis>(inner).is_none());
        assert!(analyses.get_cached::<DummyAnalysis>(outer).is_some());
        assert!(analyses.get_cached::<DummyAnalysis>(unrelated).is_some());
        assert_eq!(analyses.len(), 2);
        assert!(analyses.dependencies.is_empty());
        assert!(analyses.dependents.is_empty());
    }

    #[test]
    fn clear_removes_dependency_metadata() {
        let _counters = lock_counters();
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        let _ = analyses.get::<RootAnalysis>(&ctx, op).unwrap();
        assert!(!analyses.dependencies.is_empty());
        assert!(!analyses.dependents.is_empty());

        analyses.clear();
        assert!(analyses.is_empty());
        assert!(analyses.dependencies.is_empty());
        assert!(analyses.dependents.is_empty());
        assert!(analyses.in_progress.is_empty());
    }
}
