//! Analysis framework for TrunkIR passes.
//!
//! [`AnalysisManager`] provides lazy, cached analyses with explicit
//! invalidation semantics. Types implementing the [`Analysis`] trait are
//! computed on demand and cached keyed by `(TypeId, OpRef)`; passes that
//! mutate the IR are expected to call [`AnalysisManager::invalidate`] to
//! keep downstream consumers correct.
//!
//! Design inspired by MLIR's `AnalysisManager`. See issue #679 for context
//! and the follow-up roadmap (#680 hybrid inliner, #676 canonicalize).
//!
//! # Usage
//!
//! ```ignore
//! use trunk_ir::analysis::AnalysisManager;
//! use trunk_ir::transforms::CallGraph;
//!
//! let mut am = AnalysisManager::new();
//! let graph = am.get::<CallGraph>(ctx, module.op());
//! // `graph: Arc<CallGraph>` — safe to hold while `ctx` is mutated.
//! do_some_mutation(ctx);
//! am.invalidate::<CallGraph>(module.op());
//! ```
//!
//! # Thread-safety
//!
//! `AnalysisManager` is not shared across threads; analyses are stored as
//! `Arc<dyn Any + Send + Sync>` for future flexibility, but the manager
//! itself is single-threaded.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

use crate::context::IrContext;
use crate::refs::OpRef;

/// An analysis computable from an IR context plus a target operation
/// (typically a `core.module` op).
///
/// Analyses should be pure functions of the IR state: computing twice on
/// an unchanged context must produce equivalent results.
pub trait Analysis: Any + Send + Sync {
    /// Compute this analysis for `target` in `ctx`.
    fn compute(ctx: &IrContext, target: OpRef) -> Self
    where
        Self: Sized;
}

/// Lazy, cached storage for analyses keyed by `(TypeId, OpRef)`.
#[derive(Default)]
pub struct AnalysisManager {
    cache: HashMap<(TypeId, OpRef), Arc<dyn Any + Send + Sync>>,
}

impl AnalysisManager {
    /// Create an empty analysis manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute (or return cached) analysis `A` for `target`.
    ///
    /// Returns an `Arc` so callers may hold the result across IR
    /// mutations without keeping the manager borrowed.
    pub fn get<A: Analysis>(&mut self, ctx: &IrContext, target: OpRef) -> Arc<A> {
        let key = (TypeId::of::<A>(), target);
        let entry = self
            .cache
            .entry(key)
            .or_insert_with(|| Arc::new(A::compute(ctx, target)) as Arc<dyn Any + Send + Sync>);
        Arc::clone(entry)
            .downcast::<A>()
            .expect("analysis cache type mismatch")
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

    // A dummy analysis that records how many times `compute` was invoked.
    // Used to verify caching semantics.
    static COMPUTE_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[derive(Debug)]
    struct DummyAnalysis {
        target: OpRef,
    }

    impl Analysis for DummyAnalysis {
        fn compute(_ctx: &IrContext, target: OpRef) -> Self {
            COMPUTE_COUNT.fetch_add(1, Ordering::SeqCst);
            Self { target }
        }
    }

    #[derive(Debug)]
    struct OtherAnalysis;

    impl Analysis for OtherAnalysis {
        fn compute(_ctx: &IrContext, _target: OpRef) -> Self {
            Self
        }
    }

    #[test]
    fn get_computes_once_then_caches() {
        let (ctx, op) = test_ctx();
        let mut am = AnalysisManager::new();

        COMPUTE_COUNT.store(0, Ordering::SeqCst);
        let a1 = am.get::<DummyAnalysis>(&ctx, op);
        let a2 = am.get::<DummyAnalysis>(&ctx, op);

        assert_eq!(COMPUTE_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(a1.target, a2.target);
        assert!(Arc::ptr_eq(&a1, &a2));
        assert_eq!(am.len(), 1);
    }

    #[test]
    fn invalidate_forces_recompute() {
        let (ctx, op) = test_ctx();
        let mut am = AnalysisManager::new();

        COMPUTE_COUNT.store(0, Ordering::SeqCst);
        let _a1 = am.get::<DummyAnalysis>(&ctx, op);
        am.invalidate::<DummyAnalysis>(op);
        let _a2 = am.get::<DummyAnalysis>(&ctx, op);

        assert_eq!(COMPUTE_COUNT.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn get_cached_returns_none_before_compute() {
        let (_ctx, op) = test_ctx();
        let am = AnalysisManager::new();
        assert!(am.get_cached::<DummyAnalysis>(op).is_none());
    }

    #[test]
    fn get_cached_returns_some_after_compute() {
        let (ctx, op) = test_ctx();
        let mut am = AnalysisManager::new();
        let _ = am.get::<DummyAnalysis>(&ctx, op);
        assert!(am.get_cached::<DummyAnalysis>(op).is_some());
    }

    #[test]
    fn invalidate_all_clears_all_analyses_for_target() {
        let (ctx, op) = test_ctx();
        let mut am = AnalysisManager::new();
        let _ = am.get::<DummyAnalysis>(&ctx, op);
        let _ = am.get::<OtherAnalysis>(&ctx, op);
        assert_eq!(am.len(), 2);

        am.invalidate_all(op);
        assert!(am.is_empty());
    }

    #[test]
    fn different_analyses_cached_independently() {
        let (ctx, op) = test_ctx();
        let mut am = AnalysisManager::new();
        let _ = am.get::<DummyAnalysis>(&ctx, op);
        let _ = am.get::<OtherAnalysis>(&ctx, op);
        assert_eq!(am.len(), 2);

        am.invalidate::<DummyAnalysis>(op);
        assert!(am.get_cached::<DummyAnalysis>(op).is_none());
        assert!(am.get_cached::<OtherAnalysis>(op).is_some());
    }

    #[test]
    fn clear_drops_every_entry() {
        let (ctx, op) = test_ctx();
        let mut am = AnalysisManager::new();
        let _ = am.get::<DummyAnalysis>(&ctx, op);
        let _ = am.get::<OtherAnalysis>(&ctx, op);
        am.clear();
        assert!(am.is_empty());
    }
}
