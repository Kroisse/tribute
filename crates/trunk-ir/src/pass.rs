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
//! pm.run(&mut ctx, root_module)?;
//! ```
use std::any::Any;
use std::error::Error;
use std::ops::ControlFlow;

use derive_more::{Display, Error};

use crate::context::IrContext;
use crate::dialect::core;
use crate::ops::DialectOp;
use crate::refs::OpRef;
use crate::walk::{WalkAction, walk_op};

/// A 1st-class transformation that runs on instances of [`Self::Target`].
///
/// `run` takes `&mut self` so passes can carry per-instance mutable state
/// (counters, caches, accumulated stats) across invocations on different
/// targets. The [`PassManager`] holds each pass exclusively for the
/// duration of a [`PassManager::run`] call.
pub trait Pass {
    /// Op type this pass operates on.
    ///
    /// When this matches a [`PassManager`]'s root type, the manager invokes
    /// [`run`](Self::run) once per pass on the root. Inside a nested manager
    /// (see [`PassManager::nest`]) the manager walks the root op for
    /// instances of `Target` and invokes [`run`](Self::run) on each.
    type Target: DialectOp;

    fn name(&self) -> &'static str;

    fn run(&mut self, ctx: &mut IrContext, target: Self::Target) -> PassRunResult;
}

/// Object-safe view of [`Pass`] used inside [`PassManager`] storage.
trait ErasedPass<T: DialectOp> {
    fn name(&self) -> &'static str;
    fn run(&mut self, ctx: &mut IrContext, target: T) -> PassRunResult;
}

impl<P: Pass> ErasedPass<P::Target> for P {
    fn name(&self) -> &'static str {
        Pass::name(self)
    }
    fn run(&mut self, ctx: &mut IrContext, target: P::Target) -> PassRunResult {
        Pass::run(self, ctx, target)
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
            PassErrorKind::Verification(error) => Some(error),
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
type VerifierFn = dyn Fn(&IrContext, OpRef) -> Result<(), VerifyError>;

/// Observation-only hook invoked after each pass, mirroring the verifier's
/// timing and propagation but without a result.
///
/// Receives the context, the pass name, and the target op. Intended for
/// profiling/debugging and for exercising the manager's dispatch mechanics in
/// tests without abusing the verifier (which is for IR-invariant checking).
type InstrumentFn = dyn Fn(&IrContext, &str, OpRef);

/// Post-pass hooks threaded through the dispatch tree: a checking [`VerifierFn`]
/// and an observation-only [`InstrumentFn`]. Both follow the same timing,
/// propagation, and stale-target skip rules. Copyable since it only holds
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
            self.pm.run_on_target_with(ctx, target, hooks)?;
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

/// Orchestrates a sequence of [`Pass`] instances plus nested sub-managers.
///
/// `Root` is the op type each registered pass operates on. The default
/// [`core::Module`] matches the top of a frontend-produced IR; nested
/// managers may target any [`DialectOp`].
///
/// ```text
/// PassManager<core::Module>
///   ├─ pass A (Target = core::Module)
///   ├─ pass B (Target = core::Module)
///   └─ nest::<func::Func>()
///        ├─ pass X (Target = func::Func)  ← runs once per func.func
///        └─ pass Y (Target = func::Func)
/// ```
pub struct PassManager<Root: DialectOp + 'static = core::Module> {
    passes: Vec<Box<dyn ErasedPass<Root>>>,
    nested: Vec<Box<dyn NestedRunner>>,
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
            passes: Vec::new(),
            nested: Vec::new(),
            verifier: None,
            instrumentation: None,
        }
    }

    /// Register a pass that operates on `Root` instances.
    pub fn add_pass<P>(&mut self, pass: P) -> &mut Self
    where
        P: Pass<Target = Root> + 'static,
    {
        self.passes.push(Box::new(pass));
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
        self.nested.push(Box::new(typed));
        let last = self.nested.last_mut().expect("just pushed");
        let typed = last
            .as_any_mut()
            .downcast_mut::<TypedNested<T>>()
            .expect("freshly inserted nested manager");
        &mut typed.pm
    }

    /// Register a verifier callback invoked after each pass on this manager
    /// and any nested manager. Typical use: in debug builds, install a
    /// validation routine so a broken invariant is attributed to the pass
    /// that caused it. Replaces any previously installed verifier.
    pub fn with_verifier<F>(&mut self, verifier: F) -> &mut Self
    where
        F: Fn(&IrContext, OpRef) -> Result<(), VerifyError> + 'static,
    {
        self.verifier = Some(Box::new(verifier));
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

    /// Run all registered passes (and recursively nested managers) on
    /// `target`. Pass-level ordering is registration order; nested
    /// managers run after the parent's own passes.
    pub fn run(&mut self, ctx: &mut IrContext, target: Root) -> PassResult {
        // Split-borrow the hooks from `passes`/`nested` so we can hand
        // their references down to nested runners while iterating the pass
        // vec mutably.
        let Self {
            passes,
            nested,
            verifier,
            instrumentation,
        } = self;
        let hooks = PostPassHooks {
            verifier: verifier.as_deref(),
            instrumentation: instrumentation.as_deref(),
        };
        Self::run_passes(ctx, target, passes, nested, hooks)
    }

    /// Entry point used by nested managers, threading parent-supplied hooks
    /// through the call tree.
    fn run_on_target_with(
        &mut self,
        ctx: &mut IrContext,
        target: Root,
        parent: PostPassHooks<'_>,
    ) -> PassResult {
        let Self {
            passes,
            nested,
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
        Self::run_passes(ctx, target, passes, nested, hooks)
    }

    fn run_passes(
        ctx: &mut IrContext,
        target: Root,
        passes: &mut [Box<dyn ErasedPass<Root>>],
        nested: &mut [Box<dyn NestedRunner>],
        hooks: PostPassHooks<'_>,
    ) -> PassResult {
        // Capture attachment state at entry so we can detect a pass that
        // detaches/erases its own target (parent_block went from Some→None).
        // A top-level root op (e.g. `core.module`) has no parent block by
        // design, so we only enforce the post-attached invariant when the
        // target was attached on entry.
        let pre_attached = ctx.op(target.op_ref()).parent_block.is_some();
        for pass in passes.iter_mut() {
            let span = tracing::debug_span!("pass", name = pass.name());
            let _enter = span.enter();
            pass.run(ctx, target)
                .map_err(|failure| PassError::execution(pass.name(), failure))?;
            if !target_still_alive::<Root>(ctx, target.op_ref(), pre_attached) {
                // The pass erased or retagged its own target. Subsequent
                // passes, the verifier, and nested managers would all see
                // a stale OpRef, so stop dispatch here.
                return Ok(());
            }
            if let Some(inst) = hooks.instrumentation {
                inst(ctx, pass.name(), target.op_ref());
            }
            if let Some(v) = hooks.verifier
                && let Err(e) = v(ctx, target.op_ref())
            {
                return Err(PassError::verification(pass.name(), e));
            }
        }
        let parent_op = target.op_ref();
        for n in nested.iter_mut() {
            n.run(ctx, parent_op, hooks)?;
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
    use crate::dialect::{core, func};
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
        let path = ctx.paths.intern("test.trb".to_owned());
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
        core::module(ctx, loc, Symbol::new("test"), region)
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
        let func_ty = core::func(ctx, nil_ty, []).as_type_ref();
        let op_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .attr("sym_name", Attribute::Symbol(Symbol::new(name)))
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
        fn run(&mut self, _ctx: &mut IrContext, _target: T) -> PassRunResult {
            self.order.borrow_mut().push(self.tag);
            Ok(())
        }
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

        fn run(&mut self, _ctx: &mut IrContext, _target: T) -> PassRunResult {
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
        fn run(&mut self, _ctx: &mut IrContext, _target: T) -> PassRunResult {
            self.count += 1;
            self.mirror.set(self.count);
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
        pm.run(&mut ctx, module).unwrap();

        assert_eq!(count.get(), 1);
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
        pm.run(&mut ctx, module).unwrap();

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
        pm.run(&mut ctx, module).unwrap();

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
        pm.run(&mut ctx, module).unwrap();

        assert_eq!(*order.borrow(), vec!["module", "func"]);
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
            fn run(&mut self, ctx: &mut IrContext, target: core::Module) -> PassRunResult {
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
        pm.run(&mut ctx, module).unwrap();

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
            fn run(&mut self, ctx: &mut IrContext, target: func::Func) -> PassRunResult {
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
        pm.run(&mut ctx, module).unwrap();

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
        pm.run(&mut ctx, module).unwrap();

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
        pm.add_pass(CountingPass::<core::Module>::new(dummy.clone()));
        pm.with_instrumentation(move |_ctx, _name, _op| {
            inv_clone.set(inv_clone.get() + 1);
        });
        pm.run(&mut ctx, module).unwrap();

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
            .add_pass(CountingPass::<func::Func>::new(dummy.clone()));
        pm.with_instrumentation(move |_ctx, _name, _op| {
            inv_clone.set(inv_clone.get() + 1);
        });
        pm.run(&mut ctx, module).unwrap();

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
        pm.run(&mut ctx, module).unwrap();
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
        pm.run(&mut ctx, module).unwrap();

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
        pm.with_verifier(move |_ctx, _op| {
            verifier_count_clone.set(verifier_count_clone.get() + 1);
            Ok(())
        });

        let error = pm.run(&mut ctx, module).unwrap_err();

        assert_eq!(error.pass_name(), "failing");
        assert!(matches!(error.kind(), PassErrorKind::Execution(_)));
        assert_eq!(error.to_string(), "pass `failing` failed: boom");
        assert_eq!(*order.borrow(), vec!["before", "failing"]);
        assert_eq!(instrumentation_count.get(), 1);
        assert_eq!(verifier_count.get(), 1);
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

        let error = pm.run(&mut ctx, module).unwrap_err();

        assert_eq!(error.pass_name(), "failing");
        assert_eq!(*order.borrow(), vec!["failing"]);
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
        pm.run(&mut ctx, module).unwrap();

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
            .add_pass(CountingPass::<func::Func>::new(dummy.clone()))
            .with_instrumentation(move |_ctx, _name, _op| {
                nested_clone.set(nested_clone.get() + 1);
            });
        pm.with_instrumentation(move |_ctx, _name, _op| {
            root_clone.set(root_clone.get() + 1);
        });
        pm.run(&mut ctx, module).unwrap();

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
            .add_pass(CountingPass::<func::Func>::new(dummy.clone()));
        pm.with_instrumentation(move |_ctx, name, op| {
            seen_clone.borrow_mut().push((name.to_string(), op));
        });
        pm.run(&mut ctx, module).unwrap();

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
        pm.run(&mut ctx, module).unwrap();

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
        pm.add_pass(CountingPass::<core::Module>::new(dummy.clone()));
        pm.add_pass(CountingPass::<core::Module>::new(after.clone()));
        pm.with_instrumentation(move |_ctx, _name, _op| {
            instrumentation_count_clone.set(instrumentation_count_clone.get() + 1);
        });
        pm.with_verifier(|_ctx, _op| {
            Err(VerifyError {
                message: "boom".to_string(),
            })
        });
        let error = pm.run(&mut ctx, module).unwrap_err();

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
            .add_pass(CountingPass::<func::Func>::new(dummy.clone()));
        pm.with_verifier(|_ctx, _op| {
            Err(VerifyError {
                message: "nested boom".to_string(),
            })
        });
        let error = pm.run(&mut ctx, module).unwrap_err();

        assert_eq!(error.pass_name(), "counting");
        assert_eq!(
            error.to_string(),
            "pass `counting` broke an IR invariant: nested boom"
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
        pm.with_verifier(|_ctx, _op| Ok(()));
        pm.run(&mut ctx, module).unwrap();

        // Both module passes ran, in registration order.
        assert_eq!(*order.borrow(), vec!["a", "b"]);
    }
}
