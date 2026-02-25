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
pub struct ArenaModule(pub OpRef);

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
    pub fn body(self, ctx: &IrContext) -> super::refs::RegionRef {
        ctx.op(self.0).regions[0]
    }

    /// Get all top-level operations in the module's first block.
    pub fn ops(self, ctx: &IrContext) -> Vec<OpRef> {
        let region = self.body(ctx);
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
        let region = self.body(ctx);
        ctx.region(region).blocks.first().copied()
    }
}
