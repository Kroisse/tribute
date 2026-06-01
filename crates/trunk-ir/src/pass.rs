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
//! pm.run(&mut ctx, root_module);
//! ```
use std::any::Any;
use std::ops::ControlFlow;

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

    fn run(&mut self, ctx: &mut IrContext, target: Self::Target);
}

/// Object-safe view of [`Pass`] used inside [`PassManager`] storage.
trait ErasedPass<T: DialectOp> {
    fn name(&self) -> &'static str;
    fn run(&mut self, ctx: &mut IrContext, target: T);
}

impl<P: Pass> ErasedPass<P::Target> for P {
    fn name(&self) -> &'static str {
        Pass::name(self)
    }
    fn run(&mut self, ctx: &mut IrContext, target: P::Target) {
        Pass::run(self, ctx, target)
    }
}

/// Error returned by a verifier when a pass leaves the IR in an invalid state.
///
/// The [`PassManager`] turns an `Err` into a panic that names the offending
/// pass, so the verifier itself only describes *what* is wrong (typically a
/// summary of one or more validation failures), not *how* to react.
#[derive(Debug)]
pub struct VerifyError {
    pub message: String,
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for VerifyError {}

/// Verifier callback invoked after each pass to check IR invariants.
///
/// Returns `Ok(())` when the IR is consistent, or [`VerifyError`] describing
/// the violation. In debug builds, callers typically register a checker (e.g.
/// wrapping [`crate::validation::validate_use_chains`]) so any pass that breaks
/// an invariant is blamed immediately rather than masked by a later pass; the
/// [`PassManager`] panics with the offending pass's name on `Err`.
type VerifierFn = dyn Fn(&IrContext, OpRef) -> Result<(), VerifyError>;

/// Object-safe runner that applies a typed nested manager to a parent op.
///
/// The runner walks the parent's region tree and invokes the nested
/// manager once per target-typed op. Wraps [`PassManager<T>`] for any
/// `T: DialectOp + 'static`.
trait NestedRunner: Any {
    fn run(&mut self, ctx: &mut IrContext, parent_op: OpRef, verifier: Option<&VerifierFn>);
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct TypedNested<T: DialectOp + 'static> {
    pm: PassManager<T>,
}

impl<T: DialectOp + 'static> NestedRunner for TypedNested<T> {
    fn run(&mut self, ctx: &mut IrContext, parent_op: OpRef, verifier: Option<&VerifierFn>) {
        // Collect targets fresh on each entry so passes that erase or
        // append ops don't leave stale refs in our worklist.
        let targets = collect_targets::<T>(ctx, parent_op);
        for target in targets {
            // Skip stale refs: a previous pass in this nested manager may
            // have erased the op.
            if !T::matches(ctx, target.op_ref()) {
                continue;
            }
            self.pm.run_on_target_with(ctx, target, verifier);
        }
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

    /// Run all registered passes (and recursively nested managers) on
    /// `target`. Pass-level ordering is registration order; nested
    /// managers run after the parent's own passes.
    pub fn run(&mut self, ctx: &mut IrContext, target: Root) {
        // Split-borrow `verifier` from `passes`/`nested` so we can hand
        // the verifier reference down to nested runners while iterating
        // the pass vec mutably.
        let Self {
            passes,
            nested,
            verifier,
        } = self;
        let verifier_ref: Option<&VerifierFn> = verifier.as_deref();
        Self::run_passes(ctx, target, passes, nested, verifier_ref);
    }

    /// Entry point used by nested managers, threading a parent-supplied
    /// verifier through the call tree.
    fn run_on_target_with(
        &mut self,
        ctx: &mut IrContext,
        target: Root,
        parent_verifier: Option<&VerifierFn>,
    ) {
        let Self {
            passes,
            nested,
            verifier,
        } = self;
        // A locally-installed verifier overrides any inherited one;
        // otherwise inherit. This lets a nested manager opt into a
        // stricter checker for its sub-tree without affecting siblings.
        let verifier_ref: Option<&VerifierFn> = verifier.as_deref().or(parent_verifier);
        Self::run_passes(ctx, target, passes, nested, verifier_ref);
    }

    fn run_passes(
        ctx: &mut IrContext,
        target: Root,
        passes: &mut [Box<dyn ErasedPass<Root>>],
        nested: &mut [Box<dyn NestedRunner>],
        verifier: Option<&VerifierFn>,
    ) {
        // Capture attachment state at entry so we can detect a pass that
        // detaches/erases its own target (parent_block went from Some→None).
        // A top-level root op (e.g. `core.module`) has no parent block by
        // design, so we only enforce the post-attached invariant when the
        // target was attached on entry.
        let pre_attached = ctx.op(target.op_ref()).parent_block.is_some();
        for pass in passes.iter_mut() {
            let span = tracing::debug_span!("pass", name = pass.name());
            let _enter = span.enter();
            pass.run(ctx, target);
            if !target_still_alive::<Root>(ctx, target.op_ref(), pre_attached) {
                // The pass erased or retagged its own target. Subsequent
                // passes, the verifier, and nested managers would all see
                // a stale OpRef, so stop dispatch here.
                return;
            }
            if let Some(v) = verifier
                && let Err(e) = v(ctx, target.op_ref())
            {
                panic!("pass `{}` broke an IR invariant: {}", pass.name(), e);
            }
        }
        let parent_op = target.op_ref();
        for n in nested.iter_mut() {
            n.run(ctx, parent_op, verifier);
        }
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
        fn run(&mut self, _ctx: &mut IrContext, _target: T) {
            self.order.borrow_mut().push(self.tag);
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
        fn run(&mut self, _ctx: &mut IrContext, _target: T) {
            self.count += 1;
            self.mirror.set(self.count);
        }
    }

    #[test]
    fn module_pass_runs_once() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let count = Rc::new(Cell::new(0));
        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(count.clone()));
        pm.run(&mut ctx, module);

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
        pm.run(&mut ctx, module);

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
        pm.run(&mut ctx, module);

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
        pm.run(&mut ctx, module);

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
            fn run(&mut self, ctx: &mut IrContext, target: core::Module) {
                let region = target.body(ctx);
                let block = ctx.region(region).blocks[0];
                let first_op = ctx.block(block).ops[0];
                crate::rewrite::erase_op(ctx, first_op);
            }
        }

        let count = Rc::new(Cell::new(0));
        let mut pm = PassManager::new();
        pm.add_pass(EraseFirst);
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(count.clone()));
        pm.run(&mut ctx, module);

        // One func remains after erase; counting pass sees it once.
        assert_eq!(count.get(), 1);
    }

    /// When a pass erases its own target mid-pipeline, the manager must
    /// stop dispatching on that target — subsequent passes, the verifier,
    /// and nested managers would otherwise see a stale OpRef.
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
            fn run(&mut self, ctx: &mut IrContext, target: func::Func) {
                crate::rewrite::erase_op(ctx, target.op_ref());
            }
        }

        let after_count = Rc::new(Cell::new(0));
        let verifier_inv = Rc::new(Cell::new(0));
        let v_clone = verifier_inv.clone();

        let mut pm = PassManager::new();
        pm.nest::<func::Func>()
            .add_pass(EraseSelf)
            .add_pass(CountingPass::<func::Func>::new(after_count.clone()))
            .with_verifier(move |_ctx, _op| {
                v_clone.set(v_clone.get() + 1);
                Ok(())
            });
        pm.run(&mut ctx, module);

        // EraseSelf runs once per func (f1, f2) and invalidates the target
        // each time, so the following pass and the verifier must be skipped.
        assert_eq!(after_count.get(), 0);
        assert_eq!(verifier_inv.get(), 0);
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
        pm.run(&mut ctx, module);

        // Mirror reflects the pass's internal counter after 3 calls,
        // proving `&mut self` mutation is observable across invocations.
        assert_eq!(mirror.get(), 3);
    }

    #[test]
    fn verifier_runs_after_each_module_pass() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let dummy = Rc::new(Cell::new(0));
        let invocations = Rc::new(Cell::new(0));
        let inv_clone = invocations.clone();

        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(dummy.clone()));
        pm.add_pass(CountingPass::<core::Module>::new(dummy.clone()));
        pm.with_verifier(move |_ctx, _op| {
            inv_clone.set(inv_clone.get() + 1);
            Ok(())
        });
        pm.run(&mut ctx, module);

        // Verifier fires once after each of the 2 module-level passes.
        assert_eq!(invocations.get(), 2);
    }

    #[test]
    fn verifier_propagates_to_nested_passes() {
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
        pm.with_verifier(move |_ctx, _op| {
            inv_clone.set(inv_clone.get() + 1);
            Ok(())
        });
        pm.run(&mut ctx, module);

        // 1 module pass + 2 funcs * 1 nested pass = 3 verifier calls.
        assert_eq!(invocations.get(), 3);
    }

    #[test]
    fn empty_pass_manager_is_noop() {
        // Calling `run` on a manager with no passes and no nested
        // managers must complete without panicking.
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        let mut pm: PassManager = PassManager::new();
        pm.run(&mut ctx, module);
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
        pm.run(&mut ctx, module);

        assert_eq!(*order.borrow(), vec!["a", "b", "c"]);
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
        pm.run(&mut ctx, module);

        // Each nested manager re-walks the module independently, so
        // labels are grouped by manager rather than interleaved per func.
        assert_eq!(*order.borrow(), vec!["first", "first", "second", "second"]);
    }

    #[test]
    fn nested_verifier_overrides_inherited() {
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
            .with_verifier(move |_ctx, _op| {
                nested_clone.set(nested_clone.get() + 1);
                Ok(())
            });
        pm.with_verifier(move |_ctx, _op| {
            root_clone.set(root_clone.get() + 1);
            Ok(())
        });
        pm.run(&mut ctx, module);

        // Root verifier fires only for the module-level pass (1 call),
        // because the nested manager installs its own and does not
        // inherit the root's.
        assert_eq!(root_inv.get(), 1);
        // Nested verifier fires per func target (2 calls).
        assert_eq!(nested_inv.get(), 2);
    }

    #[test]
    fn verifier_receives_target_op_refs() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);
        let f1 = append_func(&mut ctx, module, loc, "f1");
        let f2 = append_func(&mut ctx, module, loc, "f2");
        let module_op = module.op_ref();

        let dummy = Rc::new(Cell::new(0));
        let seen: Rc<RefCell<Vec<OpRef>>> = Rc::new(RefCell::new(Vec::new()));
        let seen_clone = seen.clone();

        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(dummy.clone()));
        pm.nest::<func::Func>()
            .add_pass(CountingPass::<func::Func>::new(dummy.clone()));
        pm.with_verifier(move |_ctx, op| {
            seen_clone.borrow_mut().push(op);
            Ok(())
        });
        pm.run(&mut ctx, module);

        // Module pass → verifier(module_op), then nested manager walks
        // for func ops → verifier(f1), verifier(f2).
        assert_eq!(*seen.borrow(), vec![module_op, f1, f2]);
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
        pm.run(&mut ctx, module);

        // Both passes ran (each holds its own counter; the mirror
        // reflects the most recent set, which here is the func pass's
        // first invocation).
        assert!(count.get() >= 1);
    }

    #[test]
    #[should_panic(expected = "pass `counting` broke an IR invariant: boom")]
    fn verifier_err_panics_naming_the_pass() {
        let (mut ctx, loc) = test_ctx();
        let module = empty_module(&mut ctx, loc);

        let dummy = Rc::new(Cell::new(0));
        let mut pm = PassManager::new();
        pm.add_pass(CountingPass::<core::Module>::new(dummy.clone()));
        pm.with_verifier(|_ctx, _op| {
            Err(VerifyError {
                message: "boom".to_string(),
            })
        });
        // The verifier returns Err after the pass; the PassManager must panic
        // and name the offending pass.
        pm.run(&mut ctx, module);
    }
}
