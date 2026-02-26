//! Arena-based rewrite infrastructure.
//!
//! In-place mutation + RAUW-based rewriting, replacing the Salsa-based
//! functional rebuild approach with direct arena mutations.

pub mod applicator;
pub mod conversion_target;
pub mod pattern;
pub mod rewriter;
pub mod type_converter;

pub use applicator::{ApplyResult, PatternApplicator};
pub use conversion_target::{ArenaConversionTarget, LegalityCheck};
pub use pattern::ArenaRewritePattern;
pub use rewriter::PatternRewriter;
pub use type_converter::ArenaTypeConverter;

use super::context::IrContext;
use super::refs::OpRef;

/// Thin wrapper around an `OpRef` pointing to a `core.module` operation.
///
/// Provides convenience methods for accessing module body and operations.
#[derive(Clone, Copy, Debug)]
pub struct ArenaModule(OpRef);

impl ArenaModule {
    /// Create an `ArenaModule` wrapper, verifying it points to a `core.module` op.
    pub fn new(ctx: &IrContext, op: OpRef) -> Option<Self> {
        let data = ctx.op(op);
        if data.dialect == crate::ir::Symbol::new("core")
            && data.name == crate::ir::Symbol::new("module")
        {
            Some(ArenaModule(op))
        } else {
            None
        }
    }

    /// Get the underlying `OpRef`.
    pub fn op(self) -> OpRef {
        self.0
    }

    /// Get the module's body region.
    pub fn body(self, ctx: &IrContext) -> Option<super::refs::RegionRef> {
        ctx.op(self.0).regions.first().copied()
    }

    /// Get all top-level operations in the module's first block.
    pub fn ops(self, ctx: &IrContext) -> Vec<OpRef> {
        let region = match self.body(ctx) {
            Some(r) => r,
            None => return vec![],
        };
        let blocks = &ctx.region(region).blocks;
        if blocks.is_empty() {
            return vec![];
        }
        ctx.block(blocks[0]).ops.to_vec()
    }

    /// Get the module name (from `sym_name` attribute).
    pub fn name(self, ctx: &IrContext) -> Option<crate::ir::Symbol> {
        ctx.op(self.0)
            .attributes
            .get(&crate::ir::Symbol::new("sym_name"))
            .and_then(|a| match a {
                super::types::Attribute::Symbol(s) => Some(*s),
                _ => None,
            })
    }

    /// Get the first block of the module body.
    pub fn first_block(self, ctx: &IrContext) -> Option<super::refs::BlockRef> {
        let region = self.body(ctx)?;
        ctx.region(region).blocks.first().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::*;
    use crate::ir::Symbol;
    use crate::location::Span;
    use smallvec::smallvec;

    fn test_ctx() -> (IrContext, Location) {
        let mut ctx = IrContext::new();
        let path = ctx.paths.intern("test.trb".to_owned());
        let loc = Location::new(path, Span::new(0, 0));
        (ctx, loc)
    }

    #[test]
    fn arena_module_new_rejects_non_module_op() {
        let (mut ctx, loc) = test_ctx();
        let op_data = OperationDataBuilder::new(loc, Symbol::new("func"), Symbol::new("func"))
            .build(&mut ctx);
        let op = ctx.create_op(op_data);

        assert!(ArenaModule::new(&ctx, op).is_none());
    }

    #[test]
    fn arena_module_body_returns_none_without_regions() {
        let (mut ctx, loc) = test_ctx();
        // Create a core.module op without any regions.
        let op_data = OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
            .attr("sym_name", Attribute::Symbol(Symbol::new("empty")))
            .build(&mut ctx);
        let op = ctx.create_op(op_data);

        let module = ArenaModule::new(&ctx, op).expect("should accept core.module");
        assert!(module.body(&ctx).is_none());
        assert!(module.first_block(&ctx).is_none());
        assert!(module.ops(&ctx).is_empty());
    }

    #[test]
    fn arena_module_body_returns_region() {
        let (mut ctx, loc) = test_ctx();
        let block = ctx.create_block(BlockData {
            location: loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        let region = ctx.create_region(RegionData {
            location: loc,
            blocks: smallvec![block],
            parent_op: None,
        });
        let op_data = OperationDataBuilder::new(loc, Symbol::new("core"), Symbol::new("module"))
            .attr("sym_name", Attribute::Symbol(Symbol::new("m")))
            .region(region)
            .build(&mut ctx);
        let op = ctx.create_op(op_data);

        let module = ArenaModule::new(&ctx, op).unwrap();
        assert_eq!(module.body(&ctx), Some(region));
        assert_eq!(module.first_block(&ctx), Some(block));
    }
}
