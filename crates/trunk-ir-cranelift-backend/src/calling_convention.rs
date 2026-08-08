//! Cranelift-owned calling-convention metadata.
//!
//! Languages project their callable ABI onto the `clif.calling_convention`
//! attribute before this backend lowers or emits `clif.*`.  The generic
//! backend intentionally does not know how that provenance was established.

use cranelift_codegen::isa::CallConv;
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::clif;
use trunk_ir::refs::OpRef;

use crate::{CompilationError, CompilationResult};

/// Read the explicit Cranelift calling convention for an operation.
///
/// Untagged generic TrunkIR keeps the target platform ABI for backwards
/// compatibility.  A frontend that requires a stronger contract must attach
/// and validate this metadata before reaching the generic backend.
pub(crate) fn calling_convention_for_op(
    ctx: &IrContext,
    op: OpRef,
    platform: CallConv,
) -> CompilationResult<CallConv> {
    let Some(value) = ctx
        .op(op)
        .attributes
        .get_symbol(clif::CALLING_CONVENTION_ATTR)
    else {
        return Ok(platform);
    };

    if value == Symbol::new(clif::CALLING_CONVENTION_PLATFORM) {
        Ok(platform)
    } else if value == Symbol::new(clif::CALLING_CONVENTION_TAIL) {
        Ok(CallConv::Tail)
    } else {
        Err(CompilationError::ir_validation(format!(
            "invalid {} `{value}`",
            clif::CALLING_CONVENTION_ATTR
        )))
    }
}

#[cfg(test)]
mod tests {
    use cranelift_codegen::isa::CallConv;
    use trunk_ir::OperationDataBuilder;
    use trunk_ir::Span;
    use trunk_ir::context::IrContext;
    use trunk_ir::refs::OpRef;
    use trunk_ir::types::{Attribute, Location};

    use super::*;

    fn op_with_convention(ctx: &mut IrContext, convention: &str) -> OpRef {
        let path = ctx.paths.intern("test.trb".to_owned());
        let location = Location::new(path, Span::new(0, 0));
        let data = OperationDataBuilder::new(location, Symbol::new("clif"), Symbol::new("func"))
            .attr(
                clif::CALLING_CONVENTION_ATTR,
                Attribute::Symbol(Symbol::from_dynamic(convention)),
            )
            .build(ctx);
        ctx.create_op(data)
    }

    #[test]
    fn maps_tail_and_platform_metadata_without_a_global_module_convention() {
        let mut ctx = IrContext::new();
        let tail = op_with_convention(&mut ctx, clif::CALLING_CONVENTION_TAIL);
        let platform = op_with_convention(&mut ctx, clif::CALLING_CONVENTION_PLATFORM);

        assert_eq!(
            calling_convention_for_op(&ctx, tail, CallConv::AppleAarch64).unwrap(),
            CallConv::Tail
        );
        assert_eq!(
            calling_convention_for_op(&ctx, platform, CallConv::AppleAarch64).unwrap(),
            CallConv::AppleAarch64
        );
    }

    #[test]
    fn rejects_unknown_generic_calling_convention() {
        let mut ctx = IrContext::new();
        let op = op_with_convention(&mut ctx, "unknown");
        let error = calling_convention_for_op(&ctx, op, CallConv::AppleAarch64).unwrap_err();
        assert!(error.to_string().contains("clif.calling_convention"));
    }
}
