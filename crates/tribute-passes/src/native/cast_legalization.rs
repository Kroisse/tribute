//! Legalize `core.unrealized_conversion_cast` operations for the native target.

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::conversion::UnrealizedCastConversionPattern;
use trunk_ir::dialect::core;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::rewrite::{Module, PatternApplicator};

use super::type_converter::native_type_converter;

/// Convert the result types of unrealized casts and materialize real
/// representation changes. The identities left behind are reconciled by
/// [`trunk_ir::conversion::ReconcileUnrealizedCasts`], which runs next; a
/// remaining cast is rejected by `validate_clif_ir` before emission.
pub struct LegalizeCasts;

impl Pass for LegalizeCasts {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "legalize-casts"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let (type_converter, _) = native_type_converter(ctx);
        PatternApplicator::new(type_converter)
            .add_pattern(UnrealizedCastConversionPattern)
            .apply_partial(ctx, Module::from(target));
        Ok(())
    }
}
