//! Remove the source-logical functions nothing reachable from the roots
//! references, before CPS legalization.
//!
//! The prelude and every struct's field functions are lowered whole, and class
//! specialization leaves the instances nothing selects; most programs use
//! little of either. Removing the rest keeps every later pass from converting
//! functions the target pipeline would discard anyway.
//!
//! Bodyless declarations stay: CPS legalization checks the registered compiler
//! intrinsics against them, and they cost no lowering.

use tribute_ir::dialect::tribute_control;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};

/// Removes the source-logical functions nothing reachable references.
pub struct EliminateUnreferencedSourceFunctions;

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
        trunk_ir::transforms::eliminate_dead_definitions(ctx, target.into(), |ctx, op| {
            tribute_control::Func::matches(ctx, op) && ctx.op_has_regions(op)
        });
        Ok(())
    }
}
