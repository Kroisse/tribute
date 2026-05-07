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
    fn run(&mut self, ctx: &mut IrContext, target: T);
}

impl<P: Pass> ErasedPass<P::Target> for P {
    fn run(&mut self, ctx: &mut IrContext, target: P::Target) {
        Pass::run(self, ctx, target)
    }
}

/// Object-safe runner that applies a typed nested manager to a parent op.
///
/// The runner walks the parent's region tree and invokes the nested
/// manager once per target-typed op. Wraps [`PassManager<T>`] for any
/// `T: DialectOp + 'static`.
trait NestedRunner: Any {
    fn run(&mut self, ctx: &mut IrContext, parent_op: OpRef);
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct TypedNested<T: DialectOp + 'static> {
    pm: PassManager<T>,
}

impl<T: DialectOp + 'static> NestedRunner for TypedNested<T> {
    fn run(&mut self, ctx: &mut IrContext, parent_op: OpRef) {
        // Collect targets fresh on each entry so passes that erase or
        // append ops don't leave stale refs in our worklist.
        let targets = collect_targets::<T>(ctx, parent_op);
        for target in targets {
            // Skip stale refs: a previous pass in this nested manager may
            // have erased the op.
            if !T::matches(ctx, target.op_ref()) {
                continue;
            }
            self.pm.run_on_target(ctx, target);
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
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

    /// Run all registered passes (and recursively nested managers) on
    /// `target`. Pass-level ordering is registration order; nested
    /// managers run after the parent's own passes.
    pub fn run(&mut self, ctx: &mut IrContext, target: Root) {
        self.run_on_target(ctx, target);
    }

    fn run_on_target(&mut self, ctx: &mut IrContext, target: Root) {
        for pass in &mut self.passes {
            pass.run(ctx, target);
        }
        let parent_op = target.op_ref();
        for nested in &mut self.nested {
            nested.run(ctx, parent_op);
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

    /// Append a body-less `func.func` op into the given module.
    fn append_func(ctx: &mut IrContext, module: core::Module, loc: Location, name: &'static str) {
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

        let mut pm = PassManager::new();
        pm.add_pass(Recorder::<core::Module> {
            tag: "module",
            order: order.clone(),
            _marker: std::marker::PhantomData,
        });
        pm.nest::<func::Func>().add_pass(Recorder::<func::Func> {
            tag: "func",
            order: order.clone(),
            _marker: std::marker::PhantomData,
        });
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
}
