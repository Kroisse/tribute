//! Structural classification for operations that own an optional callable body.
//!
//! Callers select the callable operation and validate its signature and binding.
//! This query requires no dialect registration and does not mutate the IR.

use crate::{BlockRef, IrContext, OpRef, RegionRef};

/// The structurally valid shapes of an optional callable body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallableBody {
    Declaration,
    Definition { region: RegionRef, entry: BlockRef },
}

/// Target-neutral reasons that a callable body has invalid topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, derive_more::Display, derive_more::Error)]
pub enum CallableBodyError {
    #[display("body has no entry block")]
    MissingEntryBlock,
    #[display("has more than one body region")]
    MultipleBodyRegions,
}

/// Classify the region topology of a caller-selected callable operation.
///
/// Entry-block contents and subsequent blocks require separate validation.
pub fn classify_callable_body(
    ctx: &IrContext,
    op: OpRef,
) -> Result<CallableBody, CallableBodyError> {
    match ctx.op(op).regions.as_slice() {
        [] => Ok(CallableBody::Declaration),
        &[region] => ctx
            .region(region)
            .blocks
            .first()
            .copied()
            .map(|entry| CallableBody::Definition { region, entry })
            .ok_or(CallableBodyError::MissingEntryBlock),
        _ => Err(CallableBodyError::MultipleBodyRegions),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smallvec::smallvec;
    use crate::{BlockData, Location, OperationDataBuilder, RegionData, Span, Symbol};

    #[test]
    fn classifies_topology_without_dialect_signature_or_binding() {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.ir".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        let data = OperationDataBuilder::new(loc, Symbol::new("test"), Symbol::new("callable"))
            .build(&mut ctx);
        let op = ctx.create_op(data);
        assert_eq!(
            classify_callable_body(&ctx, op),
            Ok(CallableBody::Declaration)
        );

        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![],
            parent_op: Some(op),
        });
        ctx.op_mut(op).regions.push(region);
        assert_eq!(
            classify_callable_body(&ctx, op),
            Err(CallableBodyError::MissingEntryBlock)
        );

        let entry = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: Some(region),
        });
        ctx.region_mut(region).blocks.push(entry);
        let definition = CallableBody::Definition { region, entry };
        assert_eq!(classify_callable_body(&ctx, op), Ok(definition));

        let next = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: Some(region),
        });
        ctx.region_mut(region).blocks.push(next);
        assert_eq!(classify_callable_body(&ctx, op), Ok(definition));

        let extra = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![],
            parent_op: Some(op),
        });
        ctx.op_mut(op).regions.push(extra);
        assert_eq!(
            classify_callable_body(&ctx, op),
            Err(CallableBodyError::MultipleBodyRegions)
        );
    }
}
