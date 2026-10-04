//! MLIR-style pass infrastructure.
//!
//! [`Pass`] is a 1st-class transformation keyed by a [`DialectOp`] target
//! type. [`PassManager`] orchestrates a sequence of passes plus nested
//! sub-managers that target descendant op types — analogous to MLIR's
//! `PassManager` / `OpPassManager`.
//!
//! Walks reuse [`crate::walk::walk_typed`] to find target ops; the manager
//! itself does not impose ordering beyond "registration order, depth-first".
//!
//! Parallel execution is intentionally omitted: [`IrContext`] is not `Send`
//! (it stores diagnostics in a `RefCell`), and a meaningful parallelization
//! story depends on the multi-thread inventory in #682.
//!
//! # Example
//!
//! ```ignore
//! let mut pm = PassManager::new();
//! pm.add_pass(MyModulePass);
//! pm.nest::<func::Func>().add_pass(MyFunctionPass);
//! pm.run(&mut ctx, root_module, &mut analyses)?;
//! ```
use std::any::Any;
use std::error::Error;
use std::ops::ControlFlow;

use derive_more::{Display, Error};

use crate::analysis::AnalysisCache;
use crate::context::IrContext;
use crate::dialect::core;
use crate::op_interface::IsolatedFromAboveOps;
use crate::ops::DialectOp;
use crate::refs::OpRef;
use crate::rewrite::Module;
use crate::validation;
use crate::walk::{WalkAction, walk_op};

/// A 1st-class transformation that runs on instances of [`Self::Target`].
///
/// `run` takes `&mut self` so passes can carry per-instance mutable state
/// (counters, caches, accumulated stats) across invocations on different
/// targets. The [`PassManager`] holds each pass exclusively for the
/// duration of a [`PassManager::run`] call.
///
/// Every pass of one run receives the same [`AnalysisCache`]. The cache
/// discards its results whenever the IR changes, so a pass may query
/// analyses without invalidating them itself.
pub trait Pass {
    /// Op type this pass operates on.
    ///
    /// When this matches a [`PassManager`]'s root type, the manager invokes
    /// [`run`](Self::run) once per pass on the root. Inside a nested manager
    /// (see [`PassManager::nest`]) the manager walks the root op for
    /// instances of `Target` and invokes [`run`](Self::run) on each.
    type Target: DialectOp;

    fn name(&self) -> &'static str;

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: Self::Target,
        analyses: &mut AnalysisCache,
    ) -> PassRunResult;
}

/// [`Pass`] adapter for a named function or closure.
pub struct FnPass<T, F> {
    name: &'static str,
    f: F,
    _target: std::marker::PhantomData<fn(T)>,
}

impl<T, F> FnPass<T, F> {
    pub fn new(name: &'static str, f: F) -> Self {
        Self {
            name,
            f,
            _target: std::marker::PhantomData,
        }
    }
}

impl<T, F> Pass for FnPass<T, F>
where
    T: DialectOp,
    F: FnMut(&mut IrContext, T, &mut AnalysisCache) -> PassRunResult,
{
    type Target = T;

    fn name(&self) -> &'static str {
        self.name
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: T,
        analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        (self.f)(ctx, target, analyses)
    }
}

/// Build a named [`Pass`] from a function or closure.
pub fn pass_fn<T, F>(name: &'static str, f: F) -> FnPass<T, F>
where
    T: DialectOp,
    F: FnMut(&mut IrContext, T, &mut AnalysisCache) -> PassRunResult,
{
    FnPass::new(name, f)
}

/// Object-safe view of [`Pass`] used inside [`PassManager`] storage.
trait ErasedPass<T: DialectOp> {
    fn name(&self) -> &'static str;
    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: T,
        analyses: &mut AnalysisCache,
    ) -> PassRunResult;
}

impl<P: Pass> ErasedPass<P::Target> for P {
    fn name(&self) -> &'static str {
        Pass::name(self)
    }
    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: P::Target,
        analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        Pass::run(self, ctx, target, analyses)
    }
}

/// Source error returned by an individual [`Pass`] before manager context is attached.
pub type PassRunError = Box<dyn Error + Send + Sync + 'static>;

/// Result returned by an individual [`Pass`] before manager context is attached.
pub type PassRunResult = Result<(), PassRunError>;

/// Error returned by a verifier when a pass leaves the IR in an invalid state.
///
/// The verifier only describes *what* is wrong. [`PassManager`] attaches the
/// offending pass name and returns a [`PassError`].
#[derive(Debug, Display, Error)]
#[display("{message}")]
pub struct VerifyError {
    pub message: String,
}

/// Stage of a pass-manager failure.
#[derive(Debug, Display)]
pub enum PassErrorKind {
    #[display("failed: {_0}")]
    Execution(PassRunError),
    #[display("broke an IR invariant: {_0}")]
    Verification(VerifyError),
    /// The IR violated an invariant before this, the manager's first entry,
    /// ran. No pass is to blame.
    #[display("received IR that already breaks an invariant: {_0}")]
    InvalidInput(VerifyError),
}

/// Error returned by [`PassManager`] with the failing pass name attached.
#[derive(Debug)]
pub struct PassError {
    pass_name: &'static str,
    kind: PassErrorKind,
}

impl PassError {
    fn execution(pass_name: &'static str, error: PassRunError) -> Self {
        Self {
            pass_name,
            kind: PassErrorKind::Execution(error),
        }
    }

    fn verification(pass_name: &'static str, error: VerifyError) -> Self {
        Self {
            pass_name,
            kind: PassErrorKind::Verification(error),
        }
    }

    fn invalid_input(pass_name: &'static str, error: VerifyError) -> Self {
        Self {
            pass_name,
            kind: PassErrorKind::InvalidInput(error),
        }
    }

    pub fn pass_name(&self) -> &'static str {
        self.pass_name
    }

    pub fn kind(&self) -> &PassErrorKind {
        &self.kind
    }
}

impl std::fmt::Display for PassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pass `{}` {}", self.pass_name, self.kind)
    }
}

impl Error for PassError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            PassErrorKind::Execution(error) => Some(error.as_ref()),
            PassErrorKind::Verification(error) | PassErrorKind::InvalidInput(error) => Some(error),
        }
    }
}

/// Result returned by pass-manager APIs.
pub type PassResult<T = ()> = Result<T, PassError>;

/// Verifier callback invoked after each pass to check IR invariants.
///
/// Returns `Ok(())` when the IR is consistent, or [`VerifyError`] describing
/// the violation. In debug builds, callers typically register a checker (e.g.
/// wrapping [`crate::validation::validate_use_chains`]) so any pass that breaks
/// an invariant is blamed immediately rather than masked by a later pass; the
/// [`PassManager`] returns a [`PassError`] with the offending pass's name.
///
/// The verifier checks the input once before the first entry, then runs only
/// after a pass that changed the IR, since a pass that left the IR unchanged
/// cannot have broken an invariant. It receives the
/// run's [`AnalysisCache`]; it does not change the IR, so analyses it computes
/// remain cached for the passes that follow.
type VerifierFn = dyn Fn(&IrContext, &mut AnalysisCache, OpRef) -> Result<(), VerifyError>;

/// Observation-only hook invoked after each pass, mirroring the verifier's
/// timing and propagation but without a result.
///
/// Receives the context, the pass name, and the target op. Intended for
/// profiling/debugging and for exercising the manager's dispatch mechanics in
/// tests without abusing the verifier (which is for IR-invariant checking).
type InstrumentFn = dyn Fn(&IrContext, &str, OpRef);

/// Post-pass hooks threaded through the dispatch tree: a checking [`VerifierFn`]
/// and an observation-only [`InstrumentFn`]. Both follow the same timing,
/// propagation, and stale-target skip rules, except that the verifier skips a
/// pass that left the IR unchanged. Copyable since it only holds
/// borrows.
#[derive(Clone, Copy, Default)]
struct PostPassHooks<'a> {
    verifier: Option<&'a VerifierFn>,
    instrumentation: Option<&'a InstrumentFn>,
}

/// Object-safe runner that applies a typed nested manager to a parent op.
///
/// The runner walks the parent's region tree and invokes the nested
/// manager once per target-typed op. Wraps [`PassManager<T>`] for any
/// `T: DialectOp + 'static`.
trait NestedRunner: Any {
    fn run(
        &mut self,
        ctx: &mut IrContext,
        parent_op: OpRef,
        analyses: &mut AnalysisCache,
        hooks: PostPassHooks<'_>,
    ) -> PassResult;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct TypedNested<T: DialectOp + 'static> {
    pm: PassManager<T>,
}

impl<T: DialectOp + 'static> NestedRunner for TypedNested<T> {
    fn run(
        &mut self,
        ctx: &mut IrContext,
        parent_op: OpRef,
        analyses: &mut AnalysisCache,
        hooks: PostPassHooks<'_>,
    ) -> PassResult {
        // Collect targets fresh on each entry so passes that erase or
        // append ops don't leave stale refs in our worklist.
        let targets = collect_targets::<T>(ctx, parent_op);
        for target in targets {
            // Skip stale refs: a previous pass in this nested manager may
            // have erased the op.
            if !T::matches(ctx, target.op_ref()) {
                continue;
            }
            ensure_nested_anchor_is_isolated(ctx, target.op_ref())?;
            self.pm.run_on_target_with(ctx, target, analyses, hooks)?;
        }
        Ok(())
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Returns whether `op` is still safe to dispatch on after a pass ran.
///
/// `pre_attached` is the value of `parent_block.is_some()` observed before
/// the pass ran. If the op was attached before the pass and is no longer
/// attached, treat it as erased — even though the arena slot stays around
/// with intact dialect/name fields.
fn target_still_alive<T: DialectOp>(ctx: &IrContext, op: OpRef, pre_attached: bool) -> bool {
    if !T::matches(ctx, op) {
        return false;
    }
    !pre_attached || ctx.op(op).parent_block.is_some()
}

fn collect_targets<T: DialectOp>(ctx: &IrContext, root: OpRef) -> Vec<T> {
    let mut found = Vec::new();
    let _ = walk_op::<()>(ctx, root, &mut |op| {
        if let Ok(t) = T::from_op(ctx, op) {
            found.push(t);
        }
        ControlFlow::Continue(WalkAction::Advance)
    });
    found
}

fn ensure_nested_anchor_is_isolated(ctx: &IrContext, op: OpRef) -> PassResult {
    if IsolatedFromAboveOps::is_isolated(ctx, op) {
        return Ok(());
    }

    let data = ctx.op(op);
    Err(PassError::verification(
        "nested-pass-manager",
        VerifyError {
            message: format!(
                "nested pass target `{}.{}` is not registered IsolatedFromAbove",
                data.dialect, data.name
            ),
        },
    ))
}

/// Verifier installed by [`PassManager::with_debug_verifier`]. Use-chain
/// consistency (#710) comes first; the schema check assumes it.
fn verify_ir_invariants(
    ctx: &IrContext,
    _analyses: &mut AnalysisCache,
    op: OpRef,
) -> Result<(), VerifyError> {
    let Some(module) = enclosing_module(ctx, op) else {
        return Ok(());
    };
    let uses = validation::validate_use_chains(ctx, module);
    let (kind, result) = if !uses.is_ok() {
        ("use-chain", uses)
    } else {
        ("schema", validation::validate_op_schemas(ctx, op))
    };
    match result.errors.first() {
        None => Ok(()),
        Some(first) => Err(VerifyError {
            message: format!(
                "{kind} regression: {} error(s); first: {first}",
                result.errors.len()
            ),
        }),
    }
}

/// The innermost `core.module` enclosing `op`, including `op` itself.
fn enclosing_module(ctx: &IrContext, mut op: OpRef) -> Option<Module> {
    loop {
        if let Some(module) = Module::new(ctx, op) {
            return Some(module);
        }
        let block = ctx.op(op).parent_block?;
        let region = ctx.block(block).parent_region?;
        op = ctx.region(region).parent_op?;
    }
}

/// One registered step of a [`PassManager`]: a pass on the root, or a nested
/// manager applied to the root's descendants.
enum Entry<Root: DialectOp> {
    Pass(Box<dyn ErasedPass<Root>>),
    Nested(Box<dyn NestedRunner>),
}

impl<Root: DialectOp> Entry<Root> {
    fn name(&self) -> &'static str {
        match self {
            Entry::Pass(pass) => pass.name(),
            Entry::Nested(_) => "nested-pass-manager",
        }
    }
}

/// Orchestrates a sequence of [`Pass`] instances plus nested sub-managers.
///
/// `Root` is the op type each registered pass operates on. The default
/// [`core::Module`] matches the top of a frontend-produced IR; nested
/// managers may target any [`DialectOp`]. Passes and nested managers run in
/// registration order, so a module pass may follow a function-level
/// sub-pipeline.
///
/// ```text
/// PassManager<core::Module>
///   ├─ pass A (Target = core::Module)
///   ├─ nest::<func::Func>()
///   │    ├─ pass X (Target = func::Func)  ← runs once per func.func
///   │    └─ pass Y (Target = func::Func)
///   └─ pass B (Target = core::Module)     ← runs after every X and Y
/// ```
pub struct PassManager<Root: DialectOp + 'static = core::Module> {
    entries: Vec<Entry<Root>>,
    verifier: Option<Box<VerifierFn>>,
    instrumentation: Option<Box<InstrumentFn>>,
}

impl<Root: DialectOp + 'static> Default for PassManager<Root> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Root: DialectOp + 'static> PassManager<Root> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            verifier: None,
            instrumentation: None,
        }
    }

    /// Register a pass that operates on `Root` instances.
    pub fn add_pass<P>(&mut self, pass: P) -> &mut Self
    where
        P: Pass<Target = Root> + 'static,
    {
        self.entries.push(Entry::Pass(Box::new(pass)));
        self
    }

    /// Create a nested manager that walks each `Root` for `T`-typed ops and
    /// applies its registered passes per match.
    ///
    /// Returns a mutable reference to the nested manager so callers can
    /// chain `.add_pass(...)` and further `.nest::<U>()` calls.
    pub fn nest<T: DialectOp + 'static>(&mut self) -> &mut PassManager<T> {
        let typed = TypedNested {
            pm: PassManager::<T>::new(),
        };
        self.entries.push(Entry::Nested(Box::new(typed)));
        let Some(Entry::Nested(last)) = self.entries.last_mut() else {
            unreachable!("just pushed a nested manager");
        };
        let typed = last
            .as_any_mut()
            .downcast_mut::<TypedNested<T>>()
            .expect("freshly inserted nested manager");
        &mut typed.pm
    }

    /// Register a verifier callback invoked after each pass on this manager
    /// and any nested manager that changed the IR. Typical use: in debug
    /// builds, install a validation routine so a broken invariant is
    /// attributed to the pass that caused it. Replaces any previously installed verifier.
    pub fn with_verifier<F>(&mut self, verifier: F) -> &mut Self
    where
        F: Fn(&IrContext, &mut AnalysisCache, OpRef) -> Result<(), VerifyError> + 'static,
    {
        self.verifier = Some(Box::new(verifier));
        self
    }

    /// In debug builds, install the standard IR-invariant verifier: use-chain
    /// consistency of the enclosing module, then the declarative schemas of
    /// the pass target's subtree. Rewrites may break these invariants
    /// temporarily, but a finished pass must restore them. Does nothing in
    /// release builds. Replaces any previously installed verifier.
    pub fn with_debug_verifier(&mut self) -> &mut Self {
        if cfg!(debug_assertions) {
            self.with_verifier(verify_ir_invariants);
        }
        self
    }

    /// Register an observation-only instrumentation callback invoked after each
    /// pass on this manager and any nested manager (same timing/propagation as
    /// the verifier). Unlike the verifier it cannot fail the run — use it for
    /// profiling/debugging. Replaces any previously installed instrumentation.
    pub fn with_instrumentation<F>(&mut self, instrumentation: F) -> &mut Self
    where
        F: Fn(&IrContext, &str, OpRef) + 'static,
    {
        self.instrumentation = Some(Box::new(instrumentation));
        self
    }

    /// Run all registered passes and nested managers on `target`, in
    /// registration order.
    ///
    /// With a verifier installed, `target` is verified once before the first
    /// entry, so IR that was already invalid is reported as
    /// [`PassErrorKind::InvalidInput`] instead of blamed on the first pass
    /// that changes it.
    ///
    /// Every pass and the verifier query `analyses`, typically the cache of
    /// the enclosing pipeline phase.
    pub fn run(
        &mut self,
        ctx: &mut IrContext,
        target: Root,
        analyses: &mut AnalysisCache,
    ) -> PassResult {
        // Split-borrow the hooks from `entries` so we can hand their
        // references down to nested runners while iterating the entries
        // mutably.
        let Self {
            entries,
            verifier,
            instrumentation,
        } = self;
        let hooks = PostPassHooks {
            verifier: verifier.as_deref(),
            instrumentation: instrumentation.as_deref(),
        };
        if let (Some(v), Some(first)) = (hooks.verifier, entries.first()) {
            v(ctx, analyses, target.op_ref())
                .map_err(|error| PassError::invalid_input(first.name(), error))?;
        }
        Self::run_entries(ctx, target, entries, analyses, hooks)
    }

    /// Entry point used by nested managers, threading parent-supplied hooks
    /// through the call tree.
    fn run_on_target_with(
        &mut self,
        ctx: &mut IrContext,
        target: Root,
        analyses: &mut AnalysisCache,
        parent: PostPassHooks<'_>,
    ) -> PassResult {
        let Self {
            entries,
            verifier,
            instrumentation,
        } = self;
        // A locally-installed hook overrides any inherited one;
        // otherwise inherit. This lets a nested manager opt into a
        // stricter checker for its sub-tree without affecting siblings.
        let hooks = PostPassHooks {
            verifier: verifier.as_deref().or(parent.verifier),
            instrumentation: instrumentation.as_deref().or(parent.instrumentation),
        };
        Self::run_entries(ctx, target, entries, analyses, hooks)
    }

    fn run_entries(
        ctx: &mut IrContext,
        target: Root,
        entries: &mut [Entry<Root>],
        analyses: &mut AnalysisCache,
        hooks: PostPassHooks<'_>,
    ) -> PassResult {
        // Capture attachment state at entry so we can detect a pass that
        // detaches/erases its own target (parent_block went from Some→None).
        // A top-level root op (e.g. `core.module`) has no parent block by
        // design, so we only enforce the post-attached invariant when the
        // target was attached on entry.
        let pre_attached = ctx.op(target.op_ref()).parent_block.is_some();
        for entry in entries.iter_mut() {
            let pass = match entry {
                Entry::Pass(pass) => pass,
                Entry::Nested(nested) => {
                    nested.run(ctx, target.op_ref(), analyses, hooks)?;
                    // A nested pass may erase or retag an ancestor as well.
                    if !target_still_alive::<Root>(ctx, target.op_ref(), pre_attached) {
                        return Ok(());
                    }
                    continue;
                }
            };
            let span = tracing::debug_span!("pass", name = pass.name());
            let _enter = span.enter();
            let before = ctx.analysis_stamp();
            pass.run(ctx, target, analyses)
                .map_err(|failure| PassError::execution(pass.name(), failure))?;
            if !target_still_alive::<Root>(ctx, target.op_ref(), pre_attached) {
                // The pass erased or retagged its own target. Later
                // entries and the verifier would all see a stale OpRef, so
                // stop dispatch here.
                return Ok(());
            }
            if let Some(inst) = hooks.instrumentation {
                inst(ctx, pass.name(), target.op_ref());
            }
            if let Some(v) = hooks.verifier
                && ctx.analysis_stamp() != before
                && let Err(e) = v(ctx, analyses, target.op_ref())
            {
                return Err(PassError::verification(pass.name(), e));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use super::*;
    use crate::context::{BlockData, OperationDataBuilder, RegionData};
    use crate::dialect::{arith, core, func};
    use crate::location::Span;
    use crate::symbol::Symbol;
    use crate::types::{Attribute, Location};
    use smallvec::smallvec;

    #[derive(Debug)]
    struct TestFailure(&'static str);

    impl std::fmt::Display for TestFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl Error for TestFailure {}

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.intern_path("test.trb");
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    /// Build an empty `core.module` with no body.
    fn empty_module(ctx: &mut IrContext, loc: Location) -> core::Module {
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        core::Module::operands()
            .sym_name(Symbol::new("test"))
            .regions(region)
            .build(ctx, loc)
    }

    /// Append a body-less `func.func` op into the given module. Returns
    /// the new op ref so tests can assert against it.
    fn append_func(
        ctx: &mut IrContext,
        module: core::Module,
        loc: Location,
        name: &'static str,
    ) -> OpRef {
        let nil_ty = core::nil(ctx).as_type_ref();
        let func_ty = func::func_sig(ctx, [], [nil_ty]).as_type_ref();
        let op_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .attr("sym_name", Attribute::String(ctx.intern_str(name)))
            .attr("type", Attribute::Type(func_ty))
            .build(ctx);
        let func_op = ctx.create_op(op_data);
        let region = module.body(ctx);
        let block = ctx.region(region).blocks[0];
        ctx.push_op(block, func_op);
        func_op
    }

    /// Pass that pushes its `tag` onto a shared order log on every run.
    struct Recorder<T: DialectOp> {
        tag: &'static str,
        order: Rc<RefCell<Vec<&'static str>>>,
        _marker: std::marker::PhantomData<T>,
    }

    impl<T: DialectOp + 'static> Pass for Recorder<T> {
        type Target = T;
        fn name(&self) -> &'static str {
            "recorder"
        }
        fn run(
            &mut self,
            ctx: &mut IrContext,
            target: T,
            _analyses: &mut AnalysisCache,
        ) -> PassRunResult {
            self.order.borrow_mut().push(self.tag);
            touch(ctx, target.op_ref());
            Ok(())
        }
    }

    /// A verifier that accepts the manager's input and fails every later check.
    fn accept_input_then_fail(
        message: &'static str,
    ) -> impl Fn(&IrContext, &mut AnalysisCache, OpRef) -> Result<(), VerifyError> {
        let checked_input = Cell::new(false);
        move |_ctx, _analyses, _op| {
            if !checked_input.replace(true) {
                return Ok(());
            }
            Err(VerifyError {
                message: message.to_string(),
            })
        }
    }

    /// Record an IR change on `op`, so the verifier runs after the pass.
    fn touch(ctx: &mut IrContext, op: OpRef) {
        ctx.op_mut(op);
    }

    fn recorder<T: DialectOp + 'static>(
        tag: &'static str,
        order: Rc<RefCell<Vec<&'static str>>>,
    ) -> Recorder<T> {
        Recorder {
            tag,
            order,
            _marker: std::marker::PhantomData,
        }
    }

    struct FailingPass<T: DialectOp> {
        order: Rc<RefCell<Vec<&'static str>>>,
        _marker: std::marker::PhantomData<T>,
    }

    impl<T: DialectOp + 'static> Pass for FailingPass<T> {
        type Target = T;

        fn name(&self) -> &'static str {
            "failing"
        }

        fn run(
            &mut self,
            _ctx: &mut IrContext,
            _target: T,
            _analyses: &mut AnalysisCache,
        ) -> PassRunResult {
            self.order.borrow_mut().push("failing");
            Err(Box::new(TestFailure("boom")))
        }
    }

    fn failing<T: DialectOp + 'static>(order: Rc<RefCell<Vec<&'static str>>>) -> FailingPass<T> {
        FailingPass {
            order,
            _marker: std::marker::PhantomData,
        }
    }

    /// Counts invocations using a directly-owned `usize` field. The
    /// shared `Rc<Cell<usize>>` mirror lets the test observe the count
    /// after the pass is consumed by [`PassManager::add_pass`].
    struct CountingPass<T: DialectOp> {
        count: usize,
        mirror: Rc<Cell<usize>>,
        _marker: std::marker::PhantomData<T>,
    }

    impl<T: DialectOp> CountingPass<T> {
        fn new(mirror: Rc<Cell<usize>>) -> Self {
            Self {
                count: 0,
                mirror,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<T: DialectOp + 'static> Pass for CountingPass<T> {
        type Target = T;
        fn name(&self) -> &'static str {
            "counting"
        }
        fn run(
            &mut self,
            ctx: &mut IrContext,
            target: T,
            _analyses: &mut AnalysisCache,
        ) -> PassRunResult {
            self.count += 1;
            self.mirror.set(self.count);
            touch(ctx, target.op_ref());
            Ok(())
        }
    }

    #[test]
    fn module_pass_runs_once() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let count = Rc::new(Cell::new(0));
        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(count.clone()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        assert_eq!(count.get(), 1);
    }

    #[test]
    fn pass_fn_registers_named_closure_pass() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let count = Rc::new(Cell::new(0));
        let count_clone = count.clone();
        let seen_name = Rc::new(RefCell::new(String::new()));
        let seen_name_clone = seen_name.clone();

        let mut pm = PassManager::new();
        pm.add_pass(pass_fn("closure-pass", move |_ctx, _target, _analyses| {
            count_clone.set(count_clone.get() + 1);
            Ok(())
        }));
        pm.with_instrumentation(move |_ctx, name, _op| {
            seen_name_clone.replace(name.to_string());
        });
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        assert_eq!(count.get(), 1);
        assert_eq!(&*seen_name.borrow(), "closure-pass");
    }

    #[test]
    fn passes_share_the_run_analysis_cache() {
        use crate::rewrite::Module;
        use crate::symbol_table::SymbolTable;

        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f");

        let observed = Rc::new(Cell::new(0));
        let observed_in_pass = observed.clone();
        let mut pm = PassManager::new();
        pm.add_pass(pass_fn(
            "compute-symbols",
            |ctx: &mut IrContext, target: core::Module, analyses: &mut AnalysisCache| {
                let module = Module::new(ctx, target.op_ref()).expect("module target");
                analyses.require::<SymbolTable>(ctx, module.op());
                Ok(())
            },
        ));
        pm.nest::<func::Func>().add_pass(pass_fn(
            "observe-symbols",
            move |ctx: &mut IrContext, target: func::Func, analyses: &mut AnalysisCache| {
                let module = ctx
                    .op(target.op_ref())
                    .parent_block
                    .and_then(|block| ctx.region(ctx.block(block).parent_region?).parent_op);
                let cached = analyses.get_cached::<SymbolTable>(ctx, module.expect("module"));
                assert!(cached.is_some(), "unchanged IR keeps the earlier analysis");
                observed_in_pass.set(observed_in_pass.get() + 1);
                Ok(())
            },
        ));
        let mut analyses = AnalysisCache::new();
        pm.run(&mut ctx, module, &mut analyses).unwrap();

        assert_eq!(observed.get(), 1);
        assert!(
            analyses
                .get_cached::<SymbolTable>(&ctx, module.op_ref())
                .is_some(),
            "the caller's cache holds the analyses of the run"
        );
    }

    #[test]
    fn verifier_analyses_stay_cached_for_later_passes() {
        use crate::rewrite::Module;
        use crate::symbol_table::SymbolTable;

        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let observed = Rc::new(Cell::new(0));
        let observed_in_pass = observed.clone();
        let mut pm = PassManager::new();
        pm.add_pass(pass_fn(
            "change",
            |ctx: &mut IrContext, target: core::Module, _analyses: &mut AnalysisCache| {
                touch(ctx, target.op_ref());
                Ok(())
            },
        ))
        .add_pass(pass_fn(
            "observe-symbols",
            move |ctx: &mut IrContext, target: core::Module, analyses: &mut AnalysisCache| {
                let cached = analyses.get_cached::<SymbolTable>(ctx, target.op_ref());
                assert!(cached.is_some(), "the verifier's analysis is still cached");
                observed_in_pass.set(observed_in_pass.get() + 1);
                Ok(())
            },
        ));
        pm.with_verifier(|ctx, analyses, op| {
            let module = Module::new(ctx, op).expect("module target");
            analyses.require::<SymbolTable>(ctx, module.op());
            Ok(())
        });
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        assert_eq!(observed.get(), 1);
    }

    #[test]
    fn nested_func_pass_runs_per_func() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");
        append_func(&mut ctx, module, loc, "f2");
        append_func(&mut ctx, module, loc, "f3");

        let count = Rc::new(Cell::new(0));
        let mut pm = PassManager::new();
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(count.clone()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        assert_eq!(count.get(), 3);
    }

    #[test]
    fn nested_pass_handles_empty_module() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let count = Rc::new(Cell::new(0));
        let mut pm = PassManager::new();
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(count.clone()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        assert_eq!(count.get(), 0);
    }

    #[test]
    fn module_passes_run_before_nested() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");

        let order: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
        let mut pm = PassManager::new();
        pm.add_pass(recorder::<core::Module>("module", order.clone()));
        pm.nest::<func::Func>()
            .add_pass(recorder::<func::Func>("func", order.clone()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        assert_eq!(*order.borrow(), vec!["module", "func"]);
    }

    #[test]
    fn passes_and_nested_managers_run_in_registration_order() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");
        append_func(&mut ctx, module, loc, "f2");

        let order: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
        let mut pm = PassManager::new();
        pm.add_pass(recorder::<core::Module>("before", order.clone()));
        pm.nest::<func::Func>()
            .add_pass(recorder::<func::Func>("func", order.clone()));
        pm.add_pass(recorder::<core::Module>("after", order.clone()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        assert_eq!(*order.borrow(), vec!["before", "func", "func", "after"]);
    }

    /// Erasing a target op mid-pipeline must not crash the manager. The
    /// nested runner re-validates each collected ref against `T::matches`.
    #[test]
    fn nested_pass_skips_erased_ops() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");
        append_func(&mut ctx, module, loc, "f2");

        struct EraseFirst;
        impl Pass for EraseFirst {
            type Target = core::Module;
            fn name(&self) -> &'static str {
                "erase-first"
            }
            fn run(
                &mut self,
                ctx: &mut IrContext,
                target: core::Module,
                _analyses: &mut AnalysisCache,
            ) -> PassRunResult {
                let region = target.body(ctx);
                let block = ctx.region(region).blocks[0];
                let first_op = ctx.block(block).ops[0];
                crate::rewrite::erase_op(ctx, first_op);
                Ok(())
            }
        }

        let count = Rc::new(Cell::new(0));
        let mut pm = PassManager::new();
        pm.add_pass(EraseFirst);
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(count.clone()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // One func remains after erase; counting pass sees it once.
        assert_eq!(count.get(), 1);
    }

    /// When a pass erases its own target mid-pipeline, the manager must
    /// stop dispatching on that target — subsequent passes, hooks, and
    /// nested managers would otherwise see a stale OpRef.
    #[test]
    fn nested_pass_skips_dispatch_when_pass_erases_own_target() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");
        append_func(&mut ctx, module, loc, "f2");

        struct EraseSelf;
        impl Pass for EraseSelf {
            type Target = func::Func;
            fn name(&self) -> &'static str {
                "erase-self"
            }
            fn run(
                &mut self,
                ctx: &mut IrContext,
                target: func::Func,
                _analyses: &mut AnalysisCache,
            ) -> PassRunResult {
                crate::rewrite::erase_op(ctx, target.op_ref());
                Ok(())
            }
        }

        let after_count = Rc::new(Cell::new(0));
        let instrument_inv = Rc::new(Cell::new(0));
        let inv_clone = instrument_inv.clone();

        let mut pm = PassManager::new();
        pm.nest::<func::Func>()
            .add_pass(EraseSelf)
            .add_pass(CountingPass::<func::Func>::new(after_count.clone()))
            .with_instrumentation(move |_ctx, _name, _op| {
                inv_clone.set(inv_clone.get() + 1);
            });
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // EraseSelf runs once per func (f1, f2) and invalidates the target
        // each time, so the following pass and the instrumentation hook must
        // be skipped.
        assert_eq!(after_count.get(), 0);
        assert_eq!(instrument_inv.get(), 0);
    }

    /// `Pass::run` takes `&mut self`, so a pass can accumulate per-instance
    /// state across multiple invocations within a single `PassManager::run`.
    #[test]
    fn pass_accumulates_state_across_invocations() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");
        append_func(&mut ctx, module, loc, "f2");
        append_func(&mut ctx, module, loc, "f3");

        let mirror = Rc::new(Cell::new(0));
        let mut pm = PassManager::new();
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(mirror.clone()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // Mirror reflects the pass's internal counter after 3 calls,
        // proving `&mut self` mutation is observable across invocations.
        assert_eq!(mirror.get(), 3);
    }

    #[test]
    fn instrumentation_runs_after_each_module_pass() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let dummy = Rc::new(Cell::new(0));
        let invocations = Rc::new(Cell::new(0));
        let inv_clone = invocations.clone();

        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(dummy.clone()));
        pm.add_pass(CountingPass::<core::Module>::new(dummy));
        pm.with_instrumentation(move |_ctx, _name, _op| {
            inv_clone.set(inv_clone.get() + 1);
        });
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // Instrumentation fires once after each of the 2 module-level passes.
        assert_eq!(invocations.get(), 2);
    }

    #[test]
    fn instrumentation_propagates_to_nested_passes() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");
        append_func(&mut ctx, module, loc, "f2");

        let dummy = Rc::new(Cell::new(0));
        let invocations = Rc::new(Cell::new(0));
        let inv_clone = invocations.clone();

        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(dummy.clone()));
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(dummy));
        pm.with_instrumentation(move |_ctx, _name, _op| {
            inv_clone.set(inv_clone.get() + 1);
        });
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // 1 module pass + 2 funcs * 1 nested pass = 3 instrumentation calls.
        assert_eq!(invocations.get(), 3);
    }

    #[test]
    fn empty_pass_manager_is_noop() {
        // Calling `run` on a manager with no passes and no nested
        // managers must complete without panicking.
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        let mut pm: PassManager = PassManager::new();
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();
    }

    #[test]
    fn multiple_passes_run_in_registration_order() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let order: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
        let mut pm = PassManager::new();
        pm.add_pass(recorder::<core::Module>("a", order.clone()));
        pm.add_pass(recorder::<core::Module>("b", order.clone()));
        pm.add_pass(recorder::<core::Module>("c", order.clone()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        assert_eq!(*order.borrow(), vec!["a", "b", "c"]);
    }

    #[test]
    fn pass_failure_stops_later_passes() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let order: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
        let instrumentation_count = Rc::new(Cell::new(0));
        let verifier_count = Rc::new(Cell::new(0));
        let instrumentation_count_clone = instrumentation_count.clone();
        let verifier_count_clone = verifier_count.clone();
        let mut pm = PassManager::new();
        pm.add_pass(recorder::<core::Module>("before", order.clone()));
        pm.add_pass(failing::<core::Module>(order.clone()));
        pm.add_pass(recorder::<core::Module>("after", order.clone()));
        pm.with_instrumentation(move |_ctx, _name, _op| {
            instrumentation_count_clone.set(instrumentation_count_clone.get() + 1);
        });
        pm.with_verifier(move |_ctx, _analyses, _op| {
            verifier_count_clone.set(verifier_count_clone.get() + 1);
            Ok(())
        });

        let error = pm
            .run(&mut ctx, module, &mut Default::default())
            .unwrap_err();

        assert_eq!(error.pass_name(), "failing");
        assert!(matches!(error.kind(), PassErrorKind::Execution(_)));
        assert_eq!(error.to_string(), "pass `failing` failed: boom");
        assert_eq!(*order.borrow(), vec!["before", "failing"]);
        assert_eq!(instrumentation_count.get(), 1);
        // The input, then the IR after `before`.
        assert_eq!(verifier_count.get(), 2);
    }

    #[test]
    fn nested_pass_failure_stops_targets_and_sibling_managers() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");
        append_func(&mut ctx, module, loc, "f2");

        let order: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
        let mut pm = PassManager::new();
        pm.nest::<func::Func>()
            .add_pass(failing::<func::Func>(order.clone()));
        pm.nest::<func::Func>()
            .add_pass(recorder::<func::Func>("sibling", order.clone()));

        let error = pm
            .run(&mut ctx, module, &mut Default::default())
            .unwrap_err();

        assert_eq!(error.pass_name(), "failing");
        assert_eq!(*order.borrow(), vec!["failing"]);
    }

    #[test]
    fn nested_pass_requires_isolated_anchor() {
        let input = r#"core.module @test {
  func.func @f(%x: core.i32) -> core.i32 {
    %y = arith.addi %x, %x : core.i32
    func.return %y
  }
}"#;
        let mut ctx = IrContext::new();
        let module = crate::parser::parse_test_module(&mut ctx, input);
        let module =
            core::Module::from_op(&ctx, module.op()).expect("test input must parse a core.module");
        let count = Rc::new(Cell::new(0));

        let mut pm = PassManager::new();
        pm.nest::<arith::Addi>()
            .add_pass(CountingPass::<arith::Addi>::new(count.clone()));

        let error = pm
            .run(&mut ctx, module, &mut Default::default())
            .unwrap_err();

        assert_eq!(error.pass_name(), "nested-pass-manager");
        assert!(matches!(error.kind(), PassErrorKind::Verification(_)));
        assert_eq!(count.get(), 0);
        assert!(
            error
                .to_string()
                .contains("nested pass target `arith.addi` is not registered IsolatedFromAbove")
        );
    }

    #[test]
    fn sibling_nested_managers_run_in_registration_order() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");
        append_func(&mut ctx, module, loc, "f2");

        let order: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
        let mut pm = PassManager::new();
        pm.nest::<func::Func>()
            .add_pass(recorder::<func::Func>("first", order.clone()));
        pm.nest::<func::Func>()
            .add_pass(recorder::<func::Func>("second", order.clone()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // Each nested manager re-walks the module independently, so
        // labels are grouped by manager rather than interleaved per func.
        assert_eq!(*order.borrow(), vec!["first", "first", "second", "second"]);
    }

    #[test]
    fn nested_instrumentation_overrides_inherited() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");
        append_func(&mut ctx, module, loc, "f2");

        let dummy = Rc::new(Cell::new(0));
        let root_inv = Rc::new(Cell::new(0));
        let nested_inv = Rc::new(Cell::new(0));
        let root_clone = root_inv.clone();
        let nested_clone = nested_inv.clone();

        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(dummy.clone()));
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(dummy))
            .with_instrumentation(move |_ctx, _name, _op| {
                nested_clone.set(nested_clone.get() + 1);
            });
        pm.with_instrumentation(move |_ctx, _name, _op| {
            root_clone.set(root_clone.get() + 1);
        });
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // Root instrumentation fires only for the module-level pass (1 call),
        // because the nested manager installs its own and does not inherit
        // the root's.
        assert_eq!(root_inv.get(), 1);
        // Nested instrumentation fires per func target (2 calls).
        assert_eq!(nested_inv.get(), 2);
    }

    #[test]
    fn instrumentation_receives_pass_name_and_target_op() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        let f1 = append_func(&mut ctx, module, loc, "f1");
        let f2 = append_func(&mut ctx, module, loc, "f2");
        let module_op = module.op_ref();

        let dummy = Rc::new(Cell::new(0));
        let seen: Rc<RefCell<Vec<(String, OpRef)>>> = Rc::new(RefCell::new(Vec::new()));
        let seen_clone = seen.clone();

        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(dummy.clone()));
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(dummy));
        pm.with_instrumentation(move |_ctx, name, op| {
            seen_clone.borrow_mut().push((name.to_string(), op));
        });
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // Module pass → hook("counting", module_op), then nested manager walks
        // for func ops → hook("counting", f1), hook("counting", f2).
        assert_eq!(
            *seen.borrow(),
            vec![
                ("counting".to_string(), module_op),
                ("counting".to_string(), f1),
                ("counting".to_string(), f2),
            ]
        );
    }

    #[test]
    fn no_verifier_is_silent() {
        // Without `with_verifier`, passes run normally and nothing
        // additional fires. We exercise both root and nested paths.
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");

        let count = Rc::new(Cell::new(0));
        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(count.clone()));
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(count.clone()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // Both passes ran (each holds its own counter; the mirror
        // reflects the most recent set, which here is the func pass's
        // first invocation).
        assert!(count.get() >= 1);
    }

    #[test]
    fn verifier_err_returns_error_naming_the_pass() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let dummy = Rc::new(Cell::new(0));
        let after = Rc::new(Cell::new(0));
        let instrumentation_count = Rc::new(Cell::new(0));
        let instrumentation_count_clone = instrumentation_count.clone();
        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(dummy));
        pm.add_pass(CountingPass::<core::Module>::new(after.clone()));
        pm.with_instrumentation(move |_ctx, _name, _op| {
            instrumentation_count_clone.set(instrumentation_count_clone.get() + 1);
        });
        pm.with_verifier(accept_input_then_fail("boom"));
        let error = pm
            .run(&mut ctx, module, &mut Default::default())
            .unwrap_err();

        assert_eq!(error.pass_name(), "counting");
        assert!(matches!(error.kind(), PassErrorKind::Verification(_)));
        assert_eq!(
            error.to_string(),
            "pass `counting` broke an IR invariant: boom"
        );
        assert_eq!(after.get(), 0);
        assert_eq!(instrumentation_count.get(), 1);
    }

    #[test]
    fn verifier_err_in_nested_pass_propagates() {
        // A verifier installed on the root propagates into nested managers and
        // still fails the run when a nested pass leaves the IR invalid.
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");

        let dummy = Rc::new(Cell::new(0));
        let mut pm = PassManager::new();
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(dummy));
        pm.with_verifier(accept_input_then_fail("nested boom"));
        let error = pm
            .run(&mut ctx, module, &mut Default::default())
            .unwrap_err();

        assert_eq!(error.pass_name(), "counting");
        assert_eq!(
            error.to_string(),
            "pass `counting` broke an IR invariant: nested boom"
        );
    }

    #[test]
    fn verifier_skips_passes_that_leave_the_ir_unchanged() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        append_func(&mut ctx, module, loc, "f1");

        let verified: Rc<RefCell<Vec<OpRef>>> = Rc::new(RefCell::new(Vec::new()));
        let verified_clone = verified.clone();
        let mut pm = PassManager::new();
        pm.add_pass(pass_fn(
            "unchanged",
            |_ctx: &mut IrContext, _target: core::Module, _analyses: &mut AnalysisCache| Ok(()),
        ))
        .add_pass(recorder::<core::Module>("change", Rc::default()));
        pm.nest::<func::Func>().add_pass(pass_fn(
            "unchanged-func",
            |_ctx: &mut IrContext, _target: func::Func, _analyses: &mut AnalysisCache| Ok(()),
        ));
        pm.with_verifier(move |_ctx, _analyses, op| {
            verified_clone.borrow_mut().push(op);
            Ok(())
        });
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // The input, then the IR after the one pass that changed it.
        assert_eq!(*verified.borrow(), vec![module.op_ref(), module.op_ref()]);
    }

    #[test]
    fn verifier_rejects_invalid_input_before_any_pass() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let order: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
        let mut pm = PassManager::new();
        pm.add_pass(recorder::<core::Module>("first", order.clone()));
        pm.with_verifier(|_ctx, _analyses, _op| {
            Err(VerifyError {
                message: "bad input".to_string(),
            })
        });
        let error = pm
            .run(&mut ctx, module, &mut Default::default())
            .unwrap_err();

        assert!(matches!(error.kind(), PassErrorKind::InvalidInput(_)));
        assert_eq!(error.pass_name(), "recorder");
        assert_eq!(
            error.to_string(),
            "pass `recorder` received IR that already breaks an invariant: bad input"
        );
        assert!(
            order.borrow().is_empty(),
            "no pass may run on invalid input"
        );
    }

    #[test]
    fn verifier_ok_lets_pipeline_proceed() {
        // An always-Ok verifier must not interfere: every pass still runs, in
        // order. (CountingPass can't prove this — its mirror only holds the
        // latest per-instance count — so use the order recorder.)
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let order: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
        let mut pm = PassManager::new();
        pm.add_pass(recorder::<core::Module>("a", order.clone()));
        pm.add_pass(recorder::<core::Module>("b", order.clone()));
        pm.with_verifier(|_ctx, _analyses, _op| Ok(()));
        pm.run(&mut ctx, module, &mut Default::default()).unwrap();

        // Both module passes ran, in registration order.
        assert_eq!(*order.borrow(), vec!["a", "b"]);
    }
}
