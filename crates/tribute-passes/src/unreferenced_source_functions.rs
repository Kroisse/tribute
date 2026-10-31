//! Remove the source-logical functions nothing reachable from the roots
//! references, before CPS legalization.
//!
//! The prelude and every struct's field functions are lowered whole, and class
//! specialization leaves the instances nothing selects; most programs use
//! little of either. Removing the rest keeps every later pass from converting
//! functions the target pipeline would discard anyway.
//!
//! Bodyless declarations stay: CPS legalization checks the registered compiler
//! intrinsics against them, and they cost no lowering. Runtime functions stay
//! for the target lowering that calls them (`new-plans/modules.md`, "생존
//! 범위"), unless nothing reachable can make it call them.

use std::ops::ControlFlow;

use tribute_ir::dialect::tribute_control;
use tribute_ir::runtime_items::{RUNTIME_ITEMS, RuntimeItem, function_entry_points};
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::walk::{WalkAction, walk_region};

/// Removes the source-logical functions nothing reachable references,
/// keeping the functions of a runtime item set.
pub struct EliminateUnreferencedSourceFunctions {
    runtime: &'static [RuntimeItem],
    only_with_evidence_operations: bool,
}

impl EliminateUnreferencedSourceFunctions {
    /// Keep the compiler's runtime functions where something reachable can
    /// lower to an `effect` operation, which target lowering turns into
    /// runtime calls.
    pub fn new() -> Self {
        Self {
            runtime: RUNTIME_ITEMS,
            only_with_evidence_operations: true,
        }
    }

    /// Keep the functions of `runtime`.
    pub fn keeping(runtime: &'static [RuntimeItem]) -> Self {
        Self {
            runtime,
            only_with_evidence_operations: false,
        }
    }
}

/// Whether shared lowering can build an `effect` operation in `module`. Only
/// a perform, a handle, or an evidence plan becomes one.
fn has_evidence_sources(ctx: &IrContext, module: core::Module) -> bool {
    walk_region(ctx, module.body(ctx), &mut |op| {
        if tribute_control::Perform::matches(ctx, op)
            || tribute_control::Handle::matches(ctx, op)
            || ctx
                .op_attrs(op)
                .contains_key(tribute_control::EVIDENCE_PLAN_ATTR)
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(WalkAction::Advance)
        }
    })
    .is_break()
}

/// Global DCE configuration that keeps the functions of `runtime` and every
/// bodyless declaration.
pub fn source_dce_config(runtime: &[RuntimeItem]) -> trunk_ir::transforms::GlobalDceConfig {
    trunk_ir::transforms::GlobalDceConfig {
        extra_entry_points: function_entry_points(runtime),
        keep_declarations: true,
        ..Default::default()
    }
}

impl Default for EliminateUnreferencedSourceFunctions {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for EliminateUnreferencedSourceFunctions {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "eliminate-unreferenced-source-functions"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _: &mut AnalysisCache,
    ) -> PassRunResult {
        let eliminate = |ctx: &mut IrContext, runtime| {
            trunk_ir::transforms::eliminate_dead_definitions_with_config(
                ctx,
                target.into(),
                source_dce_config(runtime),
            );
        };
        eliminate(ctx, self.runtime);
        // Runtime functions contain no evidence source, so the scan sees only
        // what reaches them from the roots.
        if self.only_with_evidence_operations && !has_evidence_sources(ctx, target) {
            eliminate(ctx, &[]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use trunk_ir::parser::parse_test_module;

    use super::*;

    /// A function table lives while a live reference uses it, and keeps
    /// its targets alive only while it lives, although it has no body.
    #[test]
    fn a_function_table_lives_and_keeps_its_targets_only_while_referenced() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @m {
  !f = tribute_control.func_sig<() -> core.i32>
  tribute_cps.func @main() -> core.i32 convention(direct) {
    %f = tribute_cps.func_ref {func_ref = @"used$8"} : !f
    %r = tribute_cps.call_indirect %f {convention = "direct"} : core.i32
    tribute_control.return %r
  }
  tribute_cps.func_table @"used$8" = {direct = @"used$9D"}
  tribute_cps.func @"used$9D"() -> core.i32 convention(direct) {
    %c = arith.const {value = 1} : core.i32
    tribute_control.return %c
  }
  tribute_cps.func_table @"unused$8" = {direct = @"unused$9D"}
  tribute_cps.func @"unused$9D"() -> core.i32 convention(direct) {
    %c = arith.const {value = 2} : core.i32
    tribute_control.return %c
  }
}"#,
        );

        trunk_ir::transforms::eliminate_dead_definitions_with_config(
            &mut ctx,
            module,
            source_dce_config(&[]),
        );

        let survivors: Vec<_> = module
            .ops(&ctx)
            .iter()
            .filter_map(|&op| ctx.op_attrs(op).get_str("sym_name").map(str::to_owned))
            .collect();
        assert_eq!(survivors, ["main", "used$8", "used$9D"]);
    }
}
