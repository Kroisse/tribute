//! Native passes that run the generic `trunk-ir-cranelift-backend`
//! conversions to `clif` with the native type converter.

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir_cranelift_backend::passes::{arith_to_clif, cf_to_clif, func_to_clif, mem_to_clif};

use super::type_converter::native_type_converter;

/// Lower the `func` dialect to `clif`.
pub struct FuncToClif;

impl Pass for FuncToClif {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "func-to-clif"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let (type_converter, _) = native_type_converter(ctx);
        func_to_clif::lower(ctx, target.into(), type_converter)?;
        Ok(())
    }
}

/// Lower the `cf` dialect to `clif`.
pub struct CfToClif;

impl Pass for CfToClif {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "cf-to-clif"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let (type_converter, _) = native_type_converter(ctx);
        cf_to_clif::lower(ctx, target.into(), type_converter)?;
        Ok(())
    }
}

/// Lower the `arith` dialect to `clif`.
pub struct ArithToClif;

impl Pass for ArithToClif {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "arith-to-clif"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let (type_converter, _) = native_type_converter(ctx);
        arith_to_clif::lower(ctx, target.into(), type_converter)?;
        Ok(())
    }
}

/// Lower the `mem` dialect to `clif`.
pub struct MemToClif;

impl Pass for MemToClif {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "mem-to-clif"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        let (type_converter, _) = native_type_converter(ctx);
        mem_to_clif::lower(ctx, target.into(), type_converter)?;
        Ok(())
    }
}
