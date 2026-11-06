//! The `#[verify]` hooks of `tribute_control` operations.

use super::*;

impl trunk_ir::ops::Verify for Func {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(verify_row_declarations(ctx, self.op_ref())?)
    }
}

impl trunk_ir::ops::Verify for FuncRef {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(row_args(ctx, self.op_ref()).map(drop)?)
    }
}

impl trunk_ir::ops::Verify for Instantiate {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(row_args(ctx, self.op_ref()).map(drop)?)
    }
}

impl trunk_ir::ops::Verify for Call {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_resolved_types(ctx, self.op_ref())?;
        row_args(ctx, self.op_ref())?;
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for CallIndirect {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for TailCall {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_resolved_types(ctx, self.op_ref())?;
        row_args(ctx, self.op_ref())?;
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for TailCallIndirect {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        verify_resolved_types(ctx, self.op_ref())?;
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for Resume {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Source,
        )?)
    }
}

impl trunk_ir::ops::Verify for Perform {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(verify_resolved_types(ctx, self.op_ref())?)
    }
}

impl trunk_ir::ops::Verify for Handle {
    /// One arm per binding, no operation bound twice, and the evidence plan.
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        let op = self.op_ref();
        let bindings = HandlerBinding::list(ctx, ctx.op_attrs(op).get("handlers"))?;
        let arms = self.arms(ctx).count();
        if bindings.len() != arms {
            return Err(format!("{} handlers but {arms} arms", bindings.len()).into());
        }
        let mut seen = HashSet::default();
        if let Some(binding) = bindings
            .iter()
            .find(|binding| !seen.insert((&binding.ability, binding.op_name)))
        {
            return Err(format!(
                "duplicate handler arm for {}::{}",
                binding.ability,
                ctx.str(binding.op_name)
            )
            .into());
        }
        Ok(verify_evidence_plan(ctx, op, EvidencePlanSite::Handle)?)
    }
}
