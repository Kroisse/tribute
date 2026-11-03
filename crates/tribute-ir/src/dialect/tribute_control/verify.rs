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
        if let Some(kind) = ctx.op_attrs(self.op_ref()).get_str("operation_kind")
            && !matches!(kind, "fn" | "op")
        {
            return Err(format!("operation_kind must be \"fn\" or \"op\", found {kind:?}").into());
        }
        Ok(verify_resolved_types(ctx, self.op_ref())?)
    }
}

impl trunk_ir::ops::Verify for Handler {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        match ctx.op_attrs(self.op_ref()).get_str("kind") {
            Some("fn" | "op") | None => Ok(()),
            Some(kind) => Err(format!("kind must be \"fn\" or \"op\", found {kind:?}").into()),
        }
    }
}

impl trunk_ir::ops::Verify for Handle {
    fn verify(&self, ctx: &IrContext) -> Result<(), trunk_ir::ops::VerificationError> {
        Ok(verify_evidence_plan(
            ctx,
            self.op_ref(),
            EvidencePlanSite::Handle,
        )?)
    }
}
