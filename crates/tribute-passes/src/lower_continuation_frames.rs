//! Expansion of the abstract continuation frame surface that
//! `tribute_control_to_cps` emits.

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::pass::{Pass, PassRunResult};

use crate::tribute_control_to_cps;

/// Pass-manager wrapper of [`lower_continuation_frames`].
pub struct LowerContinuationFrames;

impl Pass for LowerContinuationFrames {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-continuation-frames"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        tribute_control_to_cps::lower_continuation_frames(ctx, target.into())
            .map_err(|error| Box::new(error) as _)
    }
}
