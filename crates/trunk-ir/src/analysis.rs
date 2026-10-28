//! Analysis framework for TrunkIR passes.
//!
//! [`AnalysisCache`] provides lazy, cached analyses keyed by `(TypeId, OpRef)`.
//! While an analysis computes, [`IrContext`] records the entities it reads,
//! and the result is stamped with the outermost `IsolatedFromAbove` anchors
//! those reads belong to. A lookup reuses a result while no mutation has
//! touched those anchors and no unclassified mutation has happened; a result
//! that read anything without an anchor chain follows the whole-context
//! revision. Prerequisites are checked by result version, and a recomputed
//! prerequisite keeps its version when [`Analysis::same_as`] holds. Explicit
//! invalidation remains available for precise removal. The contract is
//! `new-plans/ir.md`, section "분석 질의와 편집 계약".
//!
//! A lookup either returns a complete cached result or an [`AnalysisError`].
//! Analysis errors describe invalid input IR or an unsupported analysis
//! boundary, not ordinary source-language diagnostics. A failed computation
//! is never inserted: a later lookup retries from scratch after the IR is
//! repaired or changed. With unchanged IR, cached results for other analysis
//! types and targets remain available. The type-erased cache can never expose
//! a partially constructed result. Unchanged invalid IR is expected to
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
//! and boundary check that needs it. The cache is short-lived — dropped
//! when the phase returns. Reusing a cache with another [`IrContext`]
//! discards the prior context's results before lookup.
//!
//! [`PassManager::run`](crate::pass::PassManager::run) hands the phase's
//! cache to every [`Pass`](crate::pass::Pass). Consumers outside a pass
//! manager take it as a parameter:
//!
//! ```ignore
//! fn check_boundary(ctx: &IrContext, m: Module, analyses: &mut AnalysisCache) {
//!     let symbols = analyses.require::<SymbolTable>(ctx, m.op());
//!     // check the module against `symbols`…
//! }
//! ```
//!
//! Since a result is reused only while nothing it read has changed, sharing
//! the cache is always safe; it saves work where consumers read unchanged IR
//! back to back, such as the checks at a phase boundary, and where a mutation
//! leaves the anchors a result read untouched. The [`AnalysisCache::scope`]
//! helper bundles a cache with a closure for a self-contained phase.
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
//! let fresh_graph = analyses.get::<CallGraph>(ctx, module.op())?;
//! // `graph` still describes the old IR; `fresh_graph` describes the new IR.
//! ```
//!
//! # Thread-safety
//!
//! `AnalysisCache` is not shared across threads; analyses are stored as
//! `Arc<dyn Any + Send + Sync>` for future flexibility, but the cache
//! itself is single-threaded.

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::any::{Any, TypeId, type_name};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::context::{IrContext, ReadFootprint};
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

    /// Whether this freshly computed result equals `previous`, the result it
    /// replaces. When it does, results that depend on `previous` stay cached
    /// (early cutoff). The default never claims equality.
    fn same_as(&self, previous: &Self) -> bool
    where
        Self: Sized,
    {
        let _ = previous;
        false
    }
}

/// An [`Analysis`] whose computation never fails for a valid target.
///
/// Implementing this trait promises that [`Analysis::compute`] returns `Ok`
/// whenever its target satisfies the analysis contract, and that the
/// analysis depends on no fallible analysis. [`AnalysisCache::require`] then
/// returns the result directly.
pub trait InfallibleAnalysis: Analysis {}

/// Read-only IR access and dependent-analysis lookup for one computation.
///
/// A context is created only while one [`Analysis::compute`] call is active.
/// Dependent lookups use its owning cache and are recorded atomically if the
/// enclosing computation succeeds.
pub struct AnalysisContext<'a> {
    ir: &'a IrContext,
    cache: &'a mut AnalysisCache,
    /// Each prerequisite read, with the version of the result returned.
    dependencies: HashMap<AnalysisKey, u64>,
}

impl<'a> AnalysisContext<'a> {
    /// The IR context being analysed. It cannot be mutated through this API.
    pub fn ir(&self) -> &IrContext {
        self.ir
    }

    /// Get a typed prerequisite through the same cache.
    pub fn get<A: Analysis>(&mut self, target: OpRef) -> Result<Arc<A>, AnalysisError> {
        let analysis = self.cache.get::<A>(self.ir, target)?;
        let key = AnalysisKey::of::<A>(target);
        self.dependencies
            .insert(key, self.cache.cache[&key].version);
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

/// What a cached result must still match to be fresh.
#[derive(Clone, Debug)]
enum Stamp {
    /// Some read had no anchor chain: the whole-context revision.
    Revision(u64),
    /// The unclassified-change revision and each directly read outermost
    /// anchor's revision.
    Anchors {
        unclassified: u64,
        anchors: SmallVec<[(OpRef, u64); 2]>,
    },
}

impl Stamp {
    fn of(ctx: &IrContext, footprint: ReadFootprint) -> Self {
        match footprint {
            ReadFootprint::WholeContext => Self::Revision(ctx.analysis_stamp().1),
            ReadFootprint::Anchors(anchors) => Self::Anchors {
                unclassified: ctx.unclassified_revision(),
                anchors: anchors
                    .into_iter()
                    .map(|anchor| (anchor, ctx.anchor_revision(anchor)))
                    .collect(),
            },
        }
    }

    fn is_fresh(&self, ctx: &IrContext) -> bool {
        match self {
            Self::Revision(revision) => ctx.analysis_stamp().1 == *revision,
            Self::Anchors {
                unclassified,
                anchors,
            } => {
                ctx.unclassified_revision() == *unclassified
                    && anchors
                        .iter()
                        .all(|&(anchor, revision)| ctx.anchor_revision(anchor) == revision)
            }
        }
    }
}

type ErasedResult = Arc<dyn Any + Send + Sync>;

/// A published result. A stale entry stays until it is recomputed, so the
/// replacement can keep its version when it equals the old result.
struct Entry {
    result: ErasedResult,
    stamp: Stamp,
    /// Changes only when a recomputation yields an unequal result;
    /// dependents record the version they read.
    version: u64,
    /// Recomputes the entry's analysis for its target through the cache.
    recompute: fn(&mut AnalysisCache, &IrContext, OpRef) -> bool,
}

fn recompute_erased<A: Analysis>(
    cache: &mut AnalysisCache,
    ctx: &IrContext,
    target: OpRef,
) -> bool {
    cache.get::<A>(ctx, target).is_ok()
}

/// Lazy, typed cache of analyses keyed by `(TypeId, OpRef)`.
///
/// See the [module docs](self) for the pipeline-scoped ownership model.
#[derive(Default)]
pub struct AnalysisCache {
    bound_identity: Option<u64>,
    cache: HashMap<AnalysisKey, Entry>,
    next_version: u64,
    /// Computed analysis -> its direct prerequisites and the versions read.
    dependencies: HashMap<AnalysisKey, HashMap<AnalysisKey, u64>>,
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
    ///     pm.run(ctx, m, analyses)?;
    ///     // more consumers sharing `analyses`…
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
        self.sync_with(ctx);
        let key = AnalysisKey::of::<A>(target);
        if self.validate(ctx, key, true) {
            return Ok(Self::downcast(&self.cache[&key].result));
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
        let revision = ctx.analysis_stamp();
        let reads_start = ctx.begin_reads();
        let (result, dependencies) = {
            let mut computation = AnalysisContext {
                ir: ctx,
                cache: self,
                dependencies: HashMap::default(),
            };
            let result = A::compute(&mut computation, target);
            (result, computation.dependencies)
        };
        let reads = ctx.end_reads(reads_start);
        assert_eq!(
            ctx.analysis_stamp(),
            revision,
            "IR changed during analysis computation"
        );
        let completed = self
            .in_progress
            .pop()
            .expect("analysis construction stack unexpectedly empty");
        assert_eq!(completed.key, key, "analysis construction stack corrupted");

        // Publish only a complete result and its complete dependency set. A
        // failure leaves neither a cache entry nor dependency metadata behind.
        let analysis = Arc::new(result?);
        let version = match self.cache.get(&key) {
            Some(previous) if analysis.same_as(&Self::downcast::<A>(&previous.result)) => {
                previous.version
            }
            _ => {
                self.next_version += 1;
                self.next_version
            }
        };
        self.replace_dependencies(key, dependencies);
        self.cache.insert(
            key,
            Entry {
                result: analysis.clone(),
                stamp: Stamp::of(ctx, ctx.read_footprint(&reads)),
                version,
                recompute: recompute_erased::<A>,
            },
        );
        Ok(analysis)
    }

    /// Compute (or return cached) infallible analysis `A` for `target`.
    ///
    /// # Panics
    ///
    /// Panics if the computation fails anyway, which breaks the
    /// [`InfallibleAnalysis`] promise of `A`.
    pub fn require<A: InfallibleAnalysis>(&mut self, ctx: &IrContext, target: OpRef) -> Arc<A> {
        self.get(ctx, target)
            .unwrap_or_else(|error| panic!("infallible analysis failed: {error}"))
    }

    /// Return the cached analysis `A` for `target` without computing it.
    ///
    /// Returns `None` unless the result and every prerequisite it depends on
    /// are still fresh; no prerequisite is recomputed.
    pub fn get_cached<A: Analysis>(&mut self, ctx: &IrContext, target: OpRef) -> Option<Arc<A>> {
        self.sync_with(ctx);
        let key = AnalysisKey::of::<A>(target);
        self.validate(ctx, key, false)
            .then(|| Self::downcast(&self.cache[&key].result))
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
        self.bound_identity = None;
    }

    /// Number of cached analyses, fresh or not (useful for diagnostics and
    /// tests).
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    fn downcast<A: Analysis>(result: &ErasedResult) -> Arc<A> {
        Arc::clone(result)
            .downcast::<A>()
            .expect("analysis cache type mismatch")
    }

    fn sync_with(&mut self, ctx: &IrContext) {
        let (identity, _) = ctx.analysis_stamp();
        if self.bound_identity == Some(identity) {
            return;
        }
        assert!(
            self.in_progress.is_empty(),
            "IR context changed during analysis computation"
        );
        self.clear();
        self.bound_identity = Some(identity);
    }

    /// Whether `key` is cached and still describes the IR: its own reads are
    /// unchanged and every prerequisite still has the version it read. With
    /// `recompute`, a stale prerequisite is recomputed first, keeping its
    /// version when the new result equals the old one.
    fn validate(&mut self, ctx: &IrContext, key: AnalysisKey, recompute: bool) -> bool {
        let Some(entry) = self.cache.get(&key) else {
            return false;
        };
        if !entry.stamp.is_fresh(ctx) {
            return false;
        }
        let prerequisites = self
            .dependencies
            .get(&key)
            .map(|prerequisites| {
                prerequisites
                    .iter()
                    .map(|(&key, &version)| (key, version))
                    .collect::<SmallVec<[_; 4]>>()
            })
            .unwrap_or_default();
        prerequisites.into_iter().all(|(prerequisite, version)| {
            self.current_version(ctx, prerequisite, recompute) == Some(version)
        })
    }

    /// The version of `key`'s result for the current IR, if it is cached
    /// fresh or, with `recompute`, recomputes successfully.
    fn current_version(
        &mut self,
        ctx: &IrContext,
        key: AnalysisKey,
        recompute: bool,
    ) -> Option<u64> {
        let recompute_fn = self.cache.get(&key)?.recompute;
        if !self.validate(ctx, key, recompute)
            && !(recompute && recompute_fn(self, ctx, key.target))
        {
            return None;
        }
        Some(self.cache[&key].version)
    }

    fn replace_dependencies(&mut self, key: AnalysisKey, dependencies: HashMap<AnalysisKey, u64>) {
        self.remove_dependencies(key);
        if dependencies.is_empty() {
            return;
        }
        for prerequisite in dependencies.keys() {
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
        for prerequisite in prerequisites.into_keys() {
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
        let mut invalidated = HashSet::default();
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

    /// Reads its target operation and nothing else.
    #[derive(Debug)]
    struct TargetReader;

    impl Analysis for TargetReader {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let _ = ctx.ir().op(target);
            Ok(Self)
        }
    }

    /// A detached operation, whose reads have no anchor chain.
    fn detached_op(ctx: &mut IrContext) -> OpRef {
        let path = ctx.intern_path("test.trb");
        let location = crate::Location::new(path, crate::Span::new(0, 0));
        let data = crate::OperationDataBuilder::new(location, "test.op").build(ctx);
        ctx.create_op(data)
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
    struct MarkerAnalysis(Option<i128>);

    impl Analysis for MarkerAnalysis {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let marker = ctx
                .ir()
                .op_attrs(target)
                .get("marker")
                .and_then(|a| a.as_i128());
            Ok(Self(marker))
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
            let body = ctx.ir().op_region(target, 0).unwrap();
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
            let body = ctx.ir().op_region(target, 0).unwrap();
            let block = ctx.ir().region(body).blocks[0];
            let first_child = ctx.ir().block(block).ops[0];
            if ctx.ir().op_attrs(first_child).get_str("sym_name") == Some("first") {
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

    /// Counts the operations of its target's subtree.
    struct SubtreeSize;

    impl Analysis for SubtreeSize {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let _ = crate::walk::walk_op::<()>(ctx.ir(), target, &mut |_| {
                std::ops::ControlFlow::Continue(crate::walk::WalkAction::Advance)
            });
            Ok(Self)
        }
    }

    /// Reads the block that holds its target, as structured-control
    /// analyses do.
    struct EnclosingBlock;

    impl Analysis for EnclosingBlock {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            if let Some(block) = ctx.ir().op(target).parent_block {
                let _ = ctx.ir().block(block).ops.last();
            }
            Ok(Self)
        }
    }

    /// Depends on [`SubtreeSize`] of its target without reading the IR.
    struct DependsOnSubtree;

    impl Analysis for DependsOnSubtree {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            ctx.get::<SubtreeSize>(target)?;
            Ok(Self)
        }
    }

    /// Reads its target, then depends on [`TargetReader`] of it, whose only
    /// read is the same operation.
    struct ReadsThenDependsOnTarget;

    impl Analysis for ReadsThenDependsOnTarget {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let _ = ctx.ir().op(target);
            ctx.get::<TargetReader>(target)?;
            Ok(Self)
        }
    }

    #[test]
    fn recorded_reads_map_to_outermost_anchors() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f() {
    func.return
  }
  func.func @g() {
    func.return
  }
}"#,
        );
        let module = module.op();
        let body = ctx.op_region(module, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let [f, g] = ctx.block(block).ops[..] else {
            panic!("expected two functions");
        };
        let f_body = ctx.op_region(f, 0).unwrap();
        let f_return = ctx.block(ctx.region(f_body).blocks[0]).ops[0];
        let path = ctx.intern_path("test.trb");
        let location = crate::Location::new(path, crate::Span::new(0, 0));
        let detached = crate::OperationDataBuilder::new(location, "test.op").build(&mut ctx);
        let detached = ctx.create_op(detached);

        let mut cache = AnalysisCache::new();
        let footprint = |cache: &AnalysisCache, key| {
            cache.cache.get(&key).map(|entry| match &entry.stamp {
                Stamp::Revision(_) => ReadFootprint::WholeContext,
                Stamp::Anchors { anchors, .. } => {
                    ReadFootprint::Anchors(anchors.iter().map(|&(anchor, _)| anchor).collect())
                }
            })
        };
        let anchors = |ops: &[OpRef]| Some(ReadFootprint::Anchors(ops.iter().copied().collect()));

        cache.get::<SubtreeSize>(&ctx, f).unwrap();
        cache.get::<SubtreeSize>(&ctx, g).unwrap();
        cache.get::<SubtreeSize>(&ctx, module).unwrap();
        cache.get::<EnclosingBlock>(&ctx, f_return).unwrap();
        cache.get::<EnclosingBlock>(&ctx, f).unwrap();
        cache.get::<SubtreeSize>(&ctx, detached).unwrap();
        cache.get::<DependsOnSubtree>(&ctx, g).unwrap();

        assert_eq!(
            footprint(&cache, AnalysisKey::of::<SubtreeSize>(f)),
            anchors(&[f])
        );
        assert_eq!(
            footprint(&cache, AnalysisKey::of::<SubtreeSize>(g)),
            anchors(&[g])
        );
        assert_eq!(
            footprint(&cache, AnalysisKey::of::<SubtreeSize>(module)),
            anchors(&[module])
        );
        assert_eq!(
            footprint(&cache, AnalysisKey::of::<EnclosingBlock>(f_return)),
            anchors(&[f])
        );
        assert_eq!(
            footprint(&cache, AnalysisKey::of::<EnclosingBlock>(f)),
            anchors(&[module])
        );
        assert_eq!(
            footprint(&cache, AnalysisKey::of::<SubtreeSize>(detached)),
            Some(ReadFootprint::WholeContext)
        );
        assert_eq!(
            footprint(&cache, AnalysisKey::of::<DependsOnSubtree>(g)),
            anchors(&[])
        );
    }

    #[test]
    fn nested_computation_keeps_a_first_read_equal_to_the_outer_last_read() {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f() {
    func.return
  }
}"#,
        );
        let body = ctx.op_region(module.op(), 0).unwrap();
        let f = ctx.block(ctx.region(body).blocks[0]).ops[0];

        let mut cache = AnalysisCache::new();
        cache.get::<ReadsThenDependsOnTarget>(&ctx, f).unwrap();
        let entry = &cache.cache[&AnalysisKey::of::<TargetReader>(f)];
        let Stamp::Anchors { anchors, .. } = &entry.stamp else {
            panic!("expected an anchored stamp");
        };
        assert_eq!(anchors[..], [(f, ctx.anchor_revision(f))]);
    }

    /// The number of operations in the target module's body, equal across
    /// recomputations that keep the count.
    #[derive(Debug, PartialEq)]
    struct FunctionCount(usize);

    static FUNCTION_COUNTS: AtomicUsize = AtomicUsize::new(0);

    impl Analysis for FunctionCount {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            FUNCTION_COUNTS.fetch_add(1, Ordering::SeqCst);
            let body = ctx.ir().op_region(target, 0).unwrap();
            let block = ctx.ir().region(body).blocks[0];
            Ok(Self(ctx.ir().block(block).ops.len()))
        }

        fn same_as(&self, previous: &Self) -> bool {
            self == previous
        }
    }

    std::thread_local! {
        static AGGREGATE_MODULE: std::cell::Cell<Option<OpRef>> = const { std::cell::Cell::new(None) };
    }

    /// Reads its target function and depends on the module's
    /// [`FunctionCount`], which the test names through `AGGREGATE_MODULE`.
    struct BodyWithCount;

    impl Analysis for BodyWithCount {
        fn compute(ctx: &mut AnalysisContext<'_>, target: OpRef) -> Result<Self, AnalysisError> {
            let _ = crate::walk::walk_op::<()>(ctx.ir(), target, &mut |_| {
                std::ops::ControlFlow::Continue(crate::walk::WalkAction::Advance)
            });
            let module = AGGREGATE_MODULE
                .get()
                .expect("test sets the aggregate module");
            ctx.get::<FunctionCount>(module)?;
            Ok(Self)
        }
    }

    struct Functions {
        ctx: IrContext,
        module: OpRef,
        f: OpRef,
        g: OpRef,
        /// `%a`, `%s`, and the return of `@f`.
        f_ops: [OpRef; 3],
        /// `%b` and the return of `@g`.
        g_ops: [OpRef; 2],
    }

    fn functions() -> Functions {
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %a = arith.const {value = 1} : core.i32
    %s = arith.addi %x, %a : core.i32
    func.return %s
  }
  func.func @g() -> core.i32 {
    %b = arith.const {value = 2} : core.i32
    func.return %b
  }
}"#,
        )
        .op();
        let body_ops = |ctx: &IrContext, op: OpRef| {
            let region = ctx.op_region(op, 0).unwrap();
            ctx.block(ctx.region(region).blocks[0]).ops.clone()
        };
        let [f, g] = body_ops(&ctx, module)[..] else {
            panic!("expected two functions");
        };
        let f_ops = body_ops(&ctx, f)[..].try_into().unwrap();
        let g_ops = body_ops(&ctx, g)[..].try_into().unwrap();
        Functions {
            ctx,
            module,
            f,
            g,
            f_ops,
            g_ops,
        }
    }

    fn is_cached<A: Analysis>(cache: &mut AnalysisCache, ctx: &IrContext, target: OpRef) -> bool {
        cache.get_cached::<A>(ctx, target).is_some()
    }

    #[test]
    fn sibling_function_results_survive_and_module_results_recompute() {
        let Functions {
            mut ctx,
            module,
            f,
            g,
            f_ops,
            ..
        } = functions();
        let mut cache = AnalysisCache::new();
        let old_g = cache.get::<SubtreeSize>(&ctx, g).unwrap();
        cache.get::<SubtreeSize>(&ctx, f).unwrap();
        cache.get::<SubtreeSize>(&ctx, module).unwrap();

        ctx.op_attrs_mut(f_ops[1])
            .insert("tag", crate::types::Attribute::Int(1));

        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, f));
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, module));
        let again_g = cache.get::<SubtreeSize>(&ctx, g).unwrap();
        assert!(Arc::ptr_eq(&old_g, &again_g));

        // A new interned type is not IR structure.
        let ty = crate::types::TypeDataBuilder::new("test.fresh").build(&mut ctx);
        ctx.intern_type(ty);
        assert!(is_cached::<SubtreeSize>(&mut cache, &ctx, g));
    }

    #[test]
    fn equal_aggregate_keeps_dependents_but_not_direct_reads() {
        let _counters = lock_counters();
        let Functions {
            mut ctx,
            module,
            f,
            g,
            f_ops,
            ..
        } = functions();
        AGGREGATE_MODULE.set(Some(module));
        let mut cache = AnalysisCache::new();
        let old_f = cache.get::<BodyWithCount>(&ctx, f).unwrap();
        let old_g = cache.get::<BodyWithCount>(&ctx, g).unwrap();

        // Editing `@f` recomputes the aggregate to an equal count: `@g`'s
        // result survives, `@f`'s own reads changed.
        FUNCTION_COUNTS.store(0, Ordering::SeqCst);
        ctx.op_attrs_mut(f_ops[1])
            .insert("tag", crate::types::Attribute::Int(1));
        assert!(Arc::ptr_eq(
            &old_g,
            &cache.get::<BodyWithCount>(&ctx, g).unwrap()
        ));
        let new_f = cache.get::<BodyWithCount>(&ctx, f).unwrap();
        assert!(!Arc::ptr_eq(&old_f, &new_f));
        assert_eq!(FUNCTION_COUNTS.load(Ordering::SeqCst), 1);

        // The same holds when the aggregate is recomputed directly first.
        ctx.op_attrs_mut(f_ops[1])
            .insert("tag", crate::types::Attribute::Int(2));
        cache.get::<FunctionCount>(&ctx, module).unwrap();
        assert!(Arc::ptr_eq(
            &old_g,
            &cache.get::<BodyWithCount>(&ctx, g).unwrap()
        ));

        // A third function changes the count, so every dependent recomputes,
        // even after a direct recomputation of the aggregate.
        let body = ctx.op_region(module, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let mut mapping = crate::ir_mapping::IrMapping::new();
        ctx.clone_op_into_block(block, g, &mut mapping);
        cache.get::<FunctionCount>(&ctx, module).unwrap();
        assert!(!Arc::ptr_eq(
            &old_g,
            &cache.get::<BodyWithCount>(&ctx, g).unwrap()
        ));
        assert!(!is_cached::<BodyWithCount>(&mut cache, &ctx, f));
    }

    #[test]
    fn moves_invalidate_old_and_new_owners_and_the_moved_anchor() {
        let Functions {
            mut ctx,
            f,
            g,
            f_ops,
            g_ops,
            ..
        } = functions();
        let mut cache = AnalysisCache::new();
        cache.get::<SubtreeSize>(&ctx, f).unwrap();
        cache.get::<SubtreeSize>(&ctx, g).unwrap();

        ctx.detach_op(g_ops[0]);
        let f_entry = ctx.op(f_ops[0]).parent_block.unwrap();
        ctx.insert_op_before(f_entry, f_ops[2], g_ops[0]);
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, f));
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, g));

        // Reading an anchor operation is recorded against the anchor, so
        // moving the anchor itself invalidates it while its sibling stays.
        cache.get::<TargetReader>(&ctx, f).unwrap();
        cache.get::<TargetReader>(&ctx, g).unwrap();
        let module_block = ctx.op(g).parent_block.unwrap();
        ctx.detach_op(g);
        ctx.push_op(module_block, g);
        assert!(!is_cached::<TargetReader>(&mut cache, &ctx, g));
        assert!(is_cached::<TargetReader>(&mut cache, &ctx, f));
    }

    #[test]
    fn use_list_changes_invalidate_definitions_and_users() {
        let Functions {
            mut ctx,
            f,
            g,
            f_ops,
            g_ops,
            ..
        } = functions();
        let mut cache = AnalysisCache::new();
        cache.get::<SubtreeSize>(&ctx, f).unwrap();
        cache.get::<SubtreeSize>(&ctx, g).unwrap();

        let a = ctx.op_result(f_ops[0], 0);
        let entry = ctx.op(f_ops[0]).parent_block.unwrap();
        let x = ctx.block_arg(entry, 0);
        ctx.replace_all_uses(a, x);
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, f));
        assert!(is_cached::<SubtreeSize>(&mut cache, &ctx, g));

        cache.get::<SubtreeSize>(&ctx, f).unwrap();
        let b = ctx.op_result(g_ops[0], 0);
        ctx.push_op_operand(g_ops[1], b);
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, g));
        assert!(is_cached::<SubtreeSize>(&mut cache, &ctx, f));

        // A detached user of `%b` changes its use-list.
        cache.get::<SubtreeSize>(&ctx, g).unwrap();
        let location = ctx.op(g).location;
        let user = crate::OperationDataBuilder::new(location, "test.user")
            .operand(b)
            .build(&mut ctx);
        let user = ctx.create_op(user);
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, g));
        assert!(is_cached::<SubtreeSize>(&mut cache, &ctx, f));

        // Removing that detached user changes it again.
        cache.get::<SubtreeSize>(&ctx, g).unwrap();
        ctx.remove_op(user);
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, g));
        assert!(is_cached::<SubtreeSize>(&mut cache, &ctx, f));
    }

    #[test]
    fn removing_a_detached_anchor_invalidates_its_results() {
        let Functions { mut ctx, f, g, .. } = functions();
        let mut cache = AnalysisCache::new();
        cache.get::<SubtreeSize>(&ctx, f).unwrap();
        ctx.detach_op(g);
        // A detached function is its own root anchor.
        cache.get::<SubtreeSize>(&ctx, g).unwrap();
        assert!(is_cached::<SubtreeSize>(&mut cache, &ctx, g));
        ctx.remove_op(g);
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, g));
        assert!(is_cached::<SubtreeSize>(&mut cache, &ctx, f));
    }

    #[test]
    fn unclassified_changes_discard_every_scoped_result() {
        let Functions {
            mut ctx,
            f,
            g,
            f_ops,
            ..
        } = functions();
        let mut cache = AnalysisCache::new();
        cache.get::<SubtreeSize>(&ctx, f).unwrap();
        cache.get::<SubtreeSize>(&ctx, g).unwrap();
        let _ = ctx.op_mut(f_ops[2]);
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, f));
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, g));

        cache.get::<SubtreeSize>(&ctx, g).unwrap();
        let ty = crate::types::TypeDataBuilder::new("test.aliased").build(&mut ctx);
        let ty = ctx.intern_type(ty);
        assert!(is_cached::<SubtreeSize>(&mut cache, &ctx, g));
        ctx.register_type_alias(crate::symbol::Symbol::new("Aliased"), ty);
        assert!(!is_cached::<SubtreeSize>(&mut cache, &ctx, g));
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
    fn mutation_invalidates_every_result_and_preserves_old_arc_snapshot() {
        let _counters = lock_counters();
        let (mut ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        let old = analyses.get::<MarkerAnalysis>(&ctx, op).unwrap();
        let _ = analyses.get::<RootAnalysis>(&ctx, op).unwrap();
        let _ = analyses.get::<OtherAnalysis>(&ctx, op).unwrap();
        assert_eq!(analyses.len(), 5);

        ctx.op_attrs_mut(op)
            .insert("marker", crate::types::Attribute::Int(7));
        assert!(analyses.get_cached::<MarkerAnalysis>(&ctx, op).is_none());
        // Results that read no IR stay fresh.
        assert!(analyses.get_cached::<RootAnalysis>(&ctx, op).is_some());
        assert!(analyses.get_cached::<OtherAnalysis>(&ctx, op).is_some());

        let fresh = analyses.get::<MarkerAnalysis>(&ctx, op).unwrap();
        assert_eq!(old.0, None);
        assert_eq!(fresh.0, Some(7));
        assert!(!Arc::ptr_eq(&old, &fresh));
        assert!(Arc::ptr_eq(
            &fresh,
            &analyses.get::<MarkerAnalysis>(&ctx, op).unwrap()
        ));
    }

    #[test]
    fn operand_and_type_setters_invalidate_only_on_changes() {
        use crate::context::{BlockArgData, BlockData, OperationDataBuilder};

        use crate::types::TypeDataBuilder;

        let (mut ctx, module) = test_ctx();
        let i32_ty = TypeDataBuilder::new("core.i32").intern(&mut ctx);
        let f64_ty = TypeDataBuilder::new("core.f64").intern(&mut ctx);
        let location = ctx.op(module).location;
        let source_data = OperationDataBuilder::new(location, "test.source")
            .result(i32_ty)
            .build(&mut ctx);
        let source_a = ctx.create_op(source_data);
        let source_data = OperationDataBuilder::new(location, "test.source")
            .result(i32_ty)
            .build(&mut ctx);
        let source_b = ctx.create_op(source_data);
        let consumer_data = OperationDataBuilder::new(location, "test.consumer")
            .operand(ctx.op_result(source_a, 0))
            .result(i32_ty)
            .build(&mut ctx);
        let consumer = ctx.create_op(consumer_data);
        let block = ctx.create_block(BlockData {
            location,
            args: vec![BlockArgData {
                ty: i32_ty,
                attrs: Default::default(),
            }],
            ops: smallvec::SmallVec::new(),
            parent_region: None,
        });
        // The detached consumer has no anchor chain, so its result follows
        // the whole-context revision.
        let mut analyses = AnalysisCache::new();
        let _ = analyses.get::<TargetReader>(&ctx, consumer).unwrap();

        let unchanged = ctx.analysis_stamp();
        ctx.set_op_operand(consumer, 0, ctx.op_result(source_a, 0));
        ctx.set_op_result_type(consumer, 0, i32_ty);
        ctx.set_block_arg_type(block, 0, i32_ty);
        assert_eq!(ctx.analysis_stamp(), unchanged);
        assert!(
            analyses
                .get_cached::<TargetReader>(&ctx, consumer)
                .is_some()
        );

        ctx.set_op_operand(consumer, 0, ctx.op_result(source_b, 0));
        assert!(
            analyses
                .get_cached::<TargetReader>(&ctx, consumer)
                .is_none()
        );
        let _ = analyses.get::<TargetReader>(&ctx, consumer).unwrap();
        ctx.set_op_result_type(consumer, 0, f64_ty);
        assert!(
            analyses
                .get_cached::<TargetReader>(&ctx, consumer)
                .is_none()
        );
        let _ = analyses.get::<TargetReader>(&ctx, consumer).unwrap();
        ctx.set_block_arg_type(block, 0, f64_ty);
        assert!(
            analyses
                .get_cached::<TargetReader>(&ctx, consumer)
                .is_none()
        );
    }

    #[test]
    fn cache_rebinds_even_when_contexts_have_equal_revisions_and_op_refs() {
        let (mut first_ctx, first_op) = test_ctx();
        let (mut second_ctx, second_op) = test_ctx();
        first_ctx
            .op_attrs_mut(first_op)
            .insert("marker", crate::types::Attribute::Int(1));
        second_ctx
            .op_attrs_mut(second_op)
            .insert("marker", crate::types::Attribute::Int(2));
        assert_eq!(first_op, second_op);
        assert_eq!(first_ctx.analysis_stamp().1, second_ctx.analysis_stamp().1);
        assert_ne!(first_ctx.analysis_stamp().0, second_ctx.analysis_stamp().0);

        let mut analyses = AnalysisCache::new();
        let first = analyses
            .get::<MarkerAnalysis>(&first_ctx, first_op)
            .unwrap();
        assert_eq!(first.0, Some(1));
        assert!(
            analyses
                .get_cached::<MarkerAnalysis>(&second_ctx, second_op)
                .is_none()
        );
        let second = analyses
            .get::<MarkerAnalysis>(&second_ctx, second_op)
            .unwrap();
        assert_eq!(second.0, Some(2));
        assert!(!Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn interning_follows_revision_contract() {
        // A detached target follows the whole-context revision.
        let (mut ctx, _) = test_ctx();
        let op = detached_op(&mut ctx);
        let mut analyses = AnalysisCache::new();
        let initial = analyses.get::<TargetReader>(&ctx, op).unwrap();
        let stamp = ctx.analysis_stamp();

        let path = ctx.intern_path("file:///same.trb");
        assert_ne!(ctx.analysis_stamp(), stamp);
        assert!(analyses.get_cached::<TargetReader>(&ctx, op).is_none());
        let after_path = analyses.get::<TargetReader>(&ctx, op).unwrap();
        assert!(!Arc::ptr_eq(&initial, &after_path));

        let unchanged = ctx.analysis_stamp();
        assert_eq!(ctx.intern_path("file:///same.trb"), path);
        assert_eq!(ctx.analysis_stamp(), unchanged);
        assert!(Arc::ptr_eq(
            &after_path,
            &analyses.get::<TargetReader>(&ctx, op).unwrap()
        ));

        let ty = crate::types::TypeDataBuilder::new("test.fresh").build(&mut ctx);
        let interned = ctx.intern_type(ty.clone());
        assert!(analyses.get_cached::<TargetReader>(&ctx, op).is_none());
        let after_type = analyses.get::<TargetReader>(&ctx, op).unwrap();
        let unchanged = ctx.analysis_stamp();
        assert_eq!(ctx.intern_type(ty), interned);
        assert_eq!(ctx.analysis_stamp(), unchanged);
        assert!(Arc::ptr_eq(
            &after_type,
            &analyses.get::<TargetReader>(&ctx, op).unwrap()
        ));

        ctx.register_type_alias(crate::symbol::Symbol::new("Alias"), interned);
        assert!(analyses.get_cached::<TargetReader>(&ctx, op).is_none());
        let _ = analyses.get::<TargetReader>(&ctx, op).unwrap();
        let unchanged = ctx.analysis_stamp();
        ctx.register_type_alias(crate::symbol::Symbol::new("Alias"), interned);
        assert_eq!(ctx.analysis_stamp(), unchanged);
    }

    #[test]
    fn detached_builder_is_not_ir_but_raw_entity_access_invalidates() {
        use crate::context::{BlockData, OperationDataBuilder, RegionData};

        let (mut ctx, op) = test_ctx();
        let location = ctx.op(op).location;
        let stamp = ctx.analysis_stamp();
        let _detached = OperationDataBuilder::new(location, "test.unused").build(&mut ctx);
        assert_eq!(ctx.analysis_stamp(), stamp);

        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: smallvec::SmallVec::new(),
            parent_region: None,
        });
        let region = ctx.create_region(RegionData {
            location,
            blocks: smallvec::smallvec![block],
            parent_op: None,
        });
        let mut analyses = AnalysisCache::new();
        let _ = analyses.get::<DummyAnalysis>(&ctx, op).unwrap();

        let _ = ctx.op_mut(op);
        assert!(analyses.get_cached::<DummyAnalysis>(&ctx, op).is_none());
        let _ = analyses.get::<DummyAnalysis>(&ctx, op).unwrap();
        let _ = ctx.block_mut(block);
        assert!(analyses.get_cached::<DummyAnalysis>(&ctx, op).is_none());
        let _ = analyses.get::<DummyAnalysis>(&ctx, op).unwrap();
        let _ = ctx.region_mut(region);
        assert!(analyses.get_cached::<DummyAnalysis>(&ctx, op).is_none());
    }

    #[test]
    fn same_pass_mutation_and_failed_pass_cannot_reuse_old_result() {
        let (mut ctx, op) = test_ctx();
        AnalysisCache::scope(&mut ctx, |ctx, analyses| {
            let before = analyses.get::<MarkerAnalysis>(ctx, op).unwrap();
            ctx.op_attrs_mut(op)
                .insert("marker", crate::types::Attribute::Int(3));
            let after = analyses.get::<MarkerAnalysis>(ctx, op).unwrap();
            assert_eq!(before.0, None);
            assert_eq!(after.0, Some(3));

            let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ctx.op_attrs_mut(op)
                    .insert("marker", crate::types::Attribute::Int(4));
                panic!("pass failed after partial mutation");
            }));
            assert!(failed.is_err());
            assert!(analyses.get_cached::<MarkerAnalysis>(ctx, op).is_none());
            assert_eq!(analyses.get::<MarkerAnalysis>(ctx, op).unwrap().0, Some(4));
        });
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
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        assert!(analyses.get_cached::<DummyAnalysis>(&ctx, op).is_none());
    }

    #[test]
    fn get_cached_returns_some_after_compute() {
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();
        let _ = analyses.get::<DummyAnalysis>(&ctx, op).unwrap();
        assert!(analyses.get_cached::<DummyAnalysis>(&ctx, op).is_some());
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
        assert!(analyses.get_cached::<DummyAnalysis>(&ctx, op).is_none());
        assert!(analyses.get_cached::<OtherAnalysis>(&ctx, op).is_some());
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
        assert!(
            analyses
                .get_cached::<NonEmptyModuleAnalysis>(&ctx, op)
                .is_none()
        );
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
            parent_op: None,
        });
        ctx.push_op_region(op, body);

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
        let body = ctx.op_region(module, 0).unwrap();
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
                .get_cached::<NonEmptyModuleAnalysis>(&ctx, empty_module)
                .is_none()
        );
        assert!(analyses.get_cached::<DummyAnalysis>(&ctx, module).is_some());
        assert!(Arc::ptr_eq(
            &module_analysis,
            &analyses
                .get_cached::<NonEmptyModuleAnalysis>(&ctx, module)
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
        assert!(analyses.get_cached::<LeafAnalysis>(&ctx, op).is_none());
        assert!(analyses.get_cached::<MiddleAnalysis>(&ctx, op).is_none());
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
        itertools::assert_equal(
            indirect_queries.iter().map(AnalysisQuery::analysis_type),
            vec![
                type_name::<IndirectCycleFirst>(),
                type_name::<IndirectCycleSecond>(),
                type_name::<IndirectCycleFirst>(),
            ],
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
        assert!(analyses.get_cached::<RootAnalysis>(&ctx, op).is_none());
        assert!(analyses.get_cached::<MiddleAnalysis>(&ctx, op).is_some());
        assert!(analyses.get_cached::<LeafAnalysis>(&ctx, op).is_some());
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
        let body = ctx.op_region(outer, 0).unwrap();
        let block = ctx.region(body).blocks[0];
        let first = ctx.block(block).ops[0];
        let second = ctx.block(block).ops[1];
        let mut analyses = AnalysisCache::new();

        let _ = analyses.get::<ChoiceDependent>(&ctx, outer).unwrap();
        assert!(
            analyses
                .get_cached::<ChoicePrerequisiteFirst>(&ctx, first)
                .is_some()
        );
        ctx.block_mut(block).ops.swap(0, 1);
        analyses.invalidate::<ChoiceDependent>(outer);
        let _ = analyses.get::<ChoiceDependent>(&ctx, outer).unwrap();

        assert!(
            analyses
                .get_cached::<ChoicePrerequisiteSecond>(&ctx, second)
                .is_some()
        );
        analyses.invalidate::<ChoicePrerequisiteFirst>(first);
        assert!(
            analyses
                .get_cached::<ChoiceDependent>(&ctx, outer)
                .is_some()
        );
        analyses.invalidate::<ChoicePrerequisiteSecond>(second);
        assert!(
            analyses
                .get_cached::<ChoiceDependent>(&ctx, outer)
                .is_none()
        );
    }

    #[test]
    fn failed_dependent_leaves_no_cache_entry_or_dependency_edge() {
        let _counters = lock_counters();
        let (ctx, op) = test_ctx();
        let mut analyses = AnalysisCache::new();

        assert!(analyses.get::<FailingDependent>(&ctx, op).is_err());
        assert!(analyses.get_cached::<FailingDependent>(&ctx, op).is_none());
        assert!(analyses.get_cached::<LeafAnalysis>(&ctx, op).is_some());
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
        let body = ctx.op_region(outer, 0).unwrap();
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
                .get_cached::<CrossTargetPrerequisite>(&ctx, inner)
                .is_none()
        );
        assert!(
            analyses
                .get_cached::<CrossTargetDependent>(&ctx, outer)
                .is_none()
        );
        assert!(analyses.get_cached::<OtherAnalysis>(&ctx, inner).is_none());
        assert!(analyses.get_cached::<DummyAnalysis>(&ctx, outer).is_some());
        assert!(
            analyses
                .get_cached::<DummyAnalysis>(&ctx, unrelated)
                .is_some()
        );
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
