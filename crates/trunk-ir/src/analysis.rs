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
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::context::IrContext;
use crate::refs::OpRef;

/// An analysis computable from an IR context plus a target operation
/// (typically a `core.module` op).
///
/// Analyses should be pure functions of the IR state: computing twice on
/// an unchanged context must produce equivalent results.
pub trait Analysis: Any + Send + Sync {
    /// Internal IR-analysis failure returned when computation cannot produce
    /// a valid result for the requested target.
    type Error: Error + Send + Sync + 'static;

    /// Compute this analysis for `target` in `ctx`.
    ///
    /// Return an error for invalid input IR or an unsupported analysis
    /// boundary. Violated caller contracts and impossible implementation
    /// invariants must panic rather than be represented as analysis failures.
    fn compute(ctx: &IrContext, target: OpRef) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

/// Internal IR-analysis failure returned by [`AnalysisCache::get`].
///
/// The wrapper records the concrete analysis type and requested target in
/// addition to preserving the analysis-specific error as its source.
#[derive(Debug)]
pub struct AnalysisError {
    analysis_type: &'static str,
    target: OpRef,
    source: Box<dyn Error + Send + Sync>,
}

impl AnalysisError {
    fn new<A: Analysis>(target: OpRef, source: A::Error) -> Self {
        Self {
            analysis_type: type_name::<A>(),
            target,
            source: Box::new(source),
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

/// Lazy, typed cache of analyses keyed by `(TypeId, OpRef)`.
///
/// See the [module docs](self) for the pipeline-scoped ownership
/// model and the single-context invariant.
#[derive(Default)]
pub struct AnalysisCache {
    cache: HashMap<(TypeId, OpRef), Arc<dyn Any + Send + Sync>>,
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
        let key = (TypeId::of::<A>(), target);
        if let Some(entry) = self.cache.get(&key) {
            return Ok(Arc::clone(entry)
                .downcast::<A>()
                .expect("analysis cache type mismatch"));
        }

        // Compute before inserting so a failure cannot publish a partial or
        // failed type-erased entry. The next lookup is therefore a retry.
        let analysis = Arc::new(
            A::compute(ctx, target).map_err(|error| AnalysisError::new::<A>(target, error))?,
        );
        let entry: Arc<dyn Any + Send + Sync> = analysis.clone();
        self.cache.insert(key, entry);
        Ok(analysis)
    }

    /// Return the cached analysis `A` for `target` without computing it.
    pub fn get_cached<A: Analysis>(&self, target: OpRef) -> Option<Arc<A>> {
        let key = (TypeId::of::<A>(), target);
        self.cache.get(&key).map(|v| {
            Arc::clone(v)
                .downcast::<A>()
                .expect("analysis cache type mismatch")
        })
    }

    /// Invalidate the cached analysis `A` for `target`, if present.
    pub fn invalidate<A: Analysis>(&mut self, target: OpRef) {
        self.cache.remove(&(TypeId::of::<A>(), target));
    }

    /// Invalidate every cached analysis for `target`.
    pub fn invalidate_all(&mut self, target: OpRef) {
        self.cache.retain(|(_, t), _| *t != target);
    }

    /// Drop every cached analysis.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Number of cached analyses (useful for diagnostics/tests).
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

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
        type Error = TestAnalysisError;

        fn compute(_ctx: &IrContext, target: OpRef) -> Result<Self, Self::Error> {
            Ok(Self { target })
        }
    }

    #[derive(Debug)]
    struct OtherAnalysis;

    impl Analysis for OtherAnalysis {
        type Error = TestAnalysisError;

        fn compute(_ctx: &IrContext, _target: OpRef) -> Result<Self, Self::Error> {
            Ok(Self)
        }
    }

    #[derive(Debug, derive_more::Display, derive_more::Error)]
    #[display("test analysis failed")]
    struct TestAnalysisError;

    static COUNTED_COMPUTES: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug)]
    struct CountedAnalysis;

    impl Analysis for CountedAnalysis {
        type Error = TestAnalysisError;

        fn compute(_ctx: &IrContext, _target: OpRef) -> Result<Self, Self::Error> {
            COUNTED_COMPUTES.fetch_add(1, Ordering::SeqCst);
            Ok(Self)
        }
    }

    #[derive(Debug)]
    struct NonEmptyModuleAnalysis;

    impl Analysis for NonEmptyModuleAnalysis {
        type Error = TestAnalysisError;

        fn compute(ctx: &IrContext, target: OpRef) -> Result<Self, Self::Error> {
            let module = crate::rewrite::Module::new(ctx, target)
                .expect("test only requests this analysis for core.module targets");
            module
                .body(ctx)
                .filter(|&body| !ctx.region(body).blocks.is_empty())
                .map(|_| Self)
                .ok_or(TestAnalysisError)
        }
    }

    #[test]
    fn get_returns_same_arc_on_cache_hit() {
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
}
