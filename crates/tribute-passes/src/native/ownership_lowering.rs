//! Plan native ownership and lower the semantic layouts it reads.
//!
//! The ownership plan describes the IR as it stands when it is built, and
//! later steps of this pass still read it after earlier steps rewrite the
//! module, so the steps share the plan as local state of one pass instead of
//! recomputing it from the changed IR.

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::pass::{Pass, PassRunError, PassRunResult};

use super::ownership_plan::{NativeOwnershipPlanOptions, build_native_ownership_plan};
use super::type_converter::native_type_converter;
use super::{adapt_closure_layout, ownership_transfers, rc_materialization, rtti, struct_to_mem};

/// Plan ownership, materialize its RC operations, declare the RTTI layouts,
/// adapt closure layouts, lower struct field accesses to `mem.struct`, and
/// lower the ownership transfers to conversions.
pub struct LowerNativeOwnership {
    pub options: NativeOwnershipPlanOptions,
}

fn step(name: &'static str, error: impl std::error::Error + Send + Sync + 'static) -> PassRunError {
    format!("{name}: {error}").into()
}

impl Pass for LowerNativeOwnership {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-native-ownership"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let module = target.into();
        let plan = build_native_ownership_plan(ctx, module, self.options, analyses)
            .map_err(|error| step("plan-ownership", error))?;
        rc_materialization::materialize(ctx, module, &plan)
            .map_err(|error| step("materialize-rc", error))?;
        // Record the planned RTTI layouts in the IR, then adapt semantic
        // closure allocations and their declaration to the native layout.
        rtti::declare_rtti_layouts(ctx, module, plan.rtti_types());
        adapt_closure_layout::lower(ctx, module);
        // Field accesses need only the structural `mem.struct` layout, and
        // variant tests only the variant's descriptor number. The nominal
        // layouts stay on allocations, which RC header lowering resolves to
        // descriptors.
        let (type_converter, _) = native_type_converter(ctx);
        struct_to_mem::lower(ctx, module, &plan, &type_converter);
        // The plan is spent, so the transfers it validated are now only
        // changes of type.
        ownership_transfers::lower(ctx, module);
        Ok(())
    }
}
