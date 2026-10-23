//! Reject a call in a `fn` handler arm that reaches a `Cps` callee once
//! class specialization has fixed the classes of an instance.
//!
//! The rules are those of `new-plans/cps-effects.md` ("`fn` arm 호출 검사"). A
//! `fn` arm runs without a continuation frame, so a callee that may suspend
//! or abort breaks the guarantee of the `fn` declaration. Typechecking
//! rejects what the arm's definition shows; this check covers the calls
//! whose convention depends on a class an instance fixes.

use std::ops::ControlFlow;

use derive_more::{Display, Error};
use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use tribute_ir::calling_convention::CallingConvention;
use tribute_ir::dialect::tribute_control::{self, FuncSig, Row};
use tribute_ir::runtime_items::{RUNTIME_ITEMS, function_entry_points};
use trunk_ir::SymbolPath;
use trunk_ir::context::IrContext;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, RegionRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::Location;
use trunk_ir::walk::{WalkAction, walk_region};

use crate::convention_classes::{
    Function, reference_callee, requirements, row_class, settle_class_variables, settle_locals,
};

/// A call in a `fn` arm that may suspend or abort.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FnArmCall {
    pub location: Location,
    pub message: String,
}

/// The calls in `fn` arms that may suspend or abort, as a pass failure.
#[derive(Debug, Display, Error)]
#[display("{}", calls.iter().map(|call| &call.message).format("; "))]
pub struct FnArmCallError {
    #[error(not(source))]
    pub calls: Vec<FnArmCall>,
}

/// Fails when a reachable instance calls a `Cps` callee from a `fn` arm.
pub struct CheckFnArmCalls;

impl trunk_ir::pass::Pass for CheckFnArmCalls {
    type Target = trunk_ir::dialect::core::Module;

    fn name(&self) -> &'static str {
        "check-fn-arm-calls"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: trunk_ir::dialect::core::Module,
        _: &mut trunk_ir::analysis::AnalysisCache,
    ) -> trunk_ir::pass::PassRunResult {
        let calls = check(ctx, target.into());
        if calls.is_empty() {
            Ok(())
        } else {
            Err(Box::new(FnArmCallError { calls }))
        }
    }
}

/// Every call in a `fn` arm of a reachable instance that may suspend or
/// abort, once per source call. An instance is reachable from the roots the
/// elimination of unreferenced definitions keeps, so a compilation without
/// them, such as a module without `main`, checks no instance here.
pub fn check(ctx: &IrContext, module: Module) -> Vec<FnArmCall> {
    let requirements = requirements(ctx, module);
    let mut functions: Vec<Function> = module
        .ops(ctx)
        .iter()
        .filter_map(|&op| Function::read(ctx, op))
        .collect();
    let positions = settle_class_variables(&mut functions);
    let by_name: HashMap<&str, &Function> =
        functions.iter().map(|f| (f.name.as_str(), f)).collect();
    let config = trunk_ir::transforms::GlobalDceConfig {
        extra_entry_points: function_entry_points(RUNTIME_ITEMS),
        ..Default::default()
    };
    let reachable =
        trunk_ir::transforms::reachable_definitions_with_config(ctx, module, &config, |ctx, op| {
            tribute_control::Func::matches(ctx, op) && ctx.op_has_regions(op)
        });
    let mut calls = Vec::new();
    for function in functions.iter().filter(|f| reachable.contains(&f.op)) {
        let Some(body) = ctx.op_region(function.op, 0) else {
            continue;
        };
        let variables = positions.get(&function.base).map_or(&[][..], Vec::as_slice);
        let claimed = if function.is_root_main() {
            vec![CallingConvention::Direct; variables.len()]
        } else {
            function
                .claimed
                .clone()
                .unwrap_or_else(|| vec![CallingConvention::Cps; variables.len()])
        };
        let classes = variables.iter().copied().zip(claimed).collect();
        let locals = settle_locals(ctx, body, &function.binders, &requirements, &classes);
        let instance = Instance {
            ctx,
            function,
            by_name: &by_name,
            requirements: &requirements,
            classes,
            locals,
        };
        let _ = walk_region::<()>(ctx, body, &mut |op| {
            if let Ok(handler) = tribute_control::Handler::from_op(ctx, op)
                && handler.kind(ctx) == "fn"
            {
                instance.check_arm(handler, &mut calls);
            }
            ControlFlow::Continue(WalkAction::Advance)
        });
    }
    // Instances cloned from one definition repeat its calls.
    calls.into_iter().unique().collect()
}

/// One reachable instance with the classes its name fixes.
struct Instance<'a> {
    ctx: &'a IrContext,
    function: &'a Function,
    by_name: &'a HashMap<&'a str, &'a Function>,
    requirements: &'a HashMap<SymbolPath, CallingConvention>,
    classes: HashMap<u32, CallingConvention>,
    locals: HashMap<u32, CallingConvention>,
}

impl Instance<'_> {
    fn class(&self, row: &Row) -> CallingConvention {
        row_class(
            row,
            &self.function.binders,
            self.requirements,
            &self.classes,
            &self.locals,
        )
    }

    fn check_arm(&self, handler: tribute_control::Handler, calls: &mut Vec<FnArmCall>) {
        let op_name = handler.op_name(self.ctx);
        self.check_region(handler.body(self.ctx), &[], &mut |call| {
            calls.push(FnArmCall {
                location: self.ctx.op(call).location,
                message: format!(
                    "`fn` handler arm for `{op_name}` cannot call a callable that may suspend \
                     or abort: in this use of `{}`, the callee's effects include an `op` \
                     operation",
                    self.function.base
                ),
            });
        });
    }

    /// Report each call in `region` whose callee may suspend past the
    /// abilities `handled` by the `handle` bodies entered inside the arm.
    fn check_region(
        &self,
        region: RegionRef,
        handled: &[SymbolPath],
        report: &mut dyn FnMut(OpRef),
    ) {
        for &block in &self.ctx.region(region).blocks {
            for &op in &self.ctx.block(block).ops {
                self.check_op(op, handled, report);
            }
        }
    }

    fn check_op(&self, op: OpRef, handled: &[SymbolPath], report: &mut dyn FnMut(OpRef)) {
        let ctx = self.ctx;
        // A lambda's body runs where it is called, under its own convention.
        if tribute_control::Lambda::matches(ctx, op) {
            return;
        }
        if let Ok(handle) = tribute_control::Handle::from_op(ctx, op) {
            let mut inner = handled.to_vec();
            let mut arms = Vec::new();
            for &block in &ctx.region(handle.handlers(ctx)).blocks {
                for &entry in &ctx.block(block).ops {
                    if let Ok(handler) = tribute_control::Handler::from_op(ctx, entry) {
                        inner.push(handler.ability(ctx).clone());
                        // A nested `fn` arm is checked as an arm of its own.
                        if handler.kind(ctx) != "fn" {
                            arms.push(handler.body(ctx));
                        }
                    }
                }
            }
            self.check_region(handle.body(ctx), &inner, report);
            self.check_region(handle.completion(ctx), handled, report);
            for arm in arms {
                self.check_region(arm, handled, report);
            }
            return;
        }
        if self.may_suspend(op, handled) {
            report(op);
        }
        for region in ctx.op_regions(op).collect::<Vec<_>>() {
            self.check_region(region, handled, report);
        }
    }

    /// Whether the call `op` reaches a `Cps` callee whose effects are not
    /// all handled by `handled`.
    fn may_suspend(&self, op: OpRef, handled: &[SymbolPath]) -> bool {
        let ctx = self.ctx;
        let direct =
            tribute_control::Call::matches(ctx, op) || tribute_control::TailCall::matches(ctx, op);
        let (convention, row) = if direct && let Some(callee) = reference_callee(ctx, op) {
            let Some(callee) = self.by_name.get(callee.as_str()) else {
                return false;
            };
            let Some(signature) = FuncSig::from_type_ref(ctx, callee.ty) else {
                return false;
            };
            // The callee's row is over its own binders; its tail stands for
            // the row the reference passes at that binder.
            let own = signature.row(ctx);
            let row_args = tribute_control::row_args(ctx, op).unwrap_or_default();
            let passed = own
                .tail
                .and_then(|tail| row_args.get(tail as usize).cloned());
            let mut instances = own.instances;
            let tail = passed.and_then(|passed| {
                instances.extend(passed.instances);
                passed.tail
            });
            (
                signature.assigned_convention(ctx),
                Row::new(instances, tail),
            )
        } else if tribute_control::CallIndirect::matches(ctx, op)
            || tribute_control::TailCallIndirect::matches(ctx, op)
        {
            let Some(&callee) = ctx.op_operands(op).first() else {
                return false;
            };
            let Some(signature) = FuncSig::from_type_ref(ctx, ctx.value_ty(callee)) else {
                return false;
            };
            (signature.assigned_convention(ctx), signature.row(ctx))
        } else {
            return false;
        };
        if convention != Some(CallingConvention::Cps) {
            return false;
        }
        if handled.is_empty() {
            return true;
        }
        let unhandled = Row::new(
            row.instances
                .into_iter()
                .filter(|instance| !handled.contains(instance)),
            row.tail,
        );
        self.class(&unhandled) == CallingConvention::Cps
    }
}
