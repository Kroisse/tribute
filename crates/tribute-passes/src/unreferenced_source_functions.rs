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
//! 범위").

use tribute_ir::dialect::tribute_control;
use tribute_ir::runtime_items::{RUNTIME_ITEMS, RuntimeItem, function_entry_points};
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};

/// Removes the source-logical functions nothing reachable references,
/// keeping the functions of a runtime item set.
pub struct EliminateUnreferencedSourceFunctions {
    runtime: &'static [RuntimeItem],
}

impl EliminateUnreferencedSourceFunctions {
    /// Keep the compiler's runtime functions.
    pub fn new() -> Self {
        Self::keeping(RUNTIME_ITEMS)
    }

    /// Keep the functions of `runtime`.
    pub fn keeping(runtime: &'static [RuntimeItem]) -> Self {
        Self { runtime }
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
        let config = trunk_ir::transforms::GlobalDceConfig {
            extra_entry_points: function_entry_points(self.runtime),
            ..Default::default()
        };
        trunk_ir::transforms::eliminate_dead_definitions_with_config(
            ctx,
            target.into(),
            config,
            |ctx, op| tribute_control::Func::matches(ctx, op) && ctx.op_has_regions(op),
        );
        Ok(())
    }
}
