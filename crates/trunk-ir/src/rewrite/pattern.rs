//! Arena rewrite pattern trait.
//!
//! Defines the interface for arena IR transformation patterns.
//! No lifetimes needed — arena refs are Copy.

use super::rewriter::PatternRewriter;
use crate::OpName;
use crate::context::IrContext;
use crate::ops::DialectOp;
use crate::refs::OpRef;

/// A pattern that can match and transform arena IR operations.
///
/// Arena IR uses `OpRef` (Copy, no lifetime), so patterns access operations
/// through the supplied context.
///
/// # Arguments
///
/// - `ctx`: Mutable reference to the IR context for querying and mutation.
/// - `op`: The operation to match against.
/// - `rewriter`: Accumulates mutations (replace, insert, erase, add_module_op).
///
/// # Return Value
///
/// Return `true` if the pattern matched and recorded mutations via the rewriter.
/// Return `false` if the pattern does not apply.
///
/// # State
///
/// `match_and_rewrite` takes `&self`: a pattern decides from the IR and the
/// read-only data it holds or borrows from the driver, so its result does not
/// depend on the order in which the applicator visits operations. Do not cache
/// facts about the IR between calls; rewrites make them stale.
///
/// A pattern changes attached IR only through `rewriter`, so that the
/// applicator offers the affected operations again. See
/// [`PatternApplicator`](super::PatternApplicator).
pub trait RewritePattern {
    /// Attempt to match and rewrite an operation.
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool;

    /// The operations this pattern can match, or `None` if it may match any
    /// operation.
    ///
    /// The applicator offers the pattern only operations with one of these
    /// names, so a pattern that declares roots never matches another
    /// operation.
    fn roots(&self) -> Option<&[OpName]> {
        None
    }

    /// Optional: return a human-readable name for debugging.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// A pattern that rewrites one operation, given as its typed wrapper.
///
/// Implement this instead of [`RewritePattern`] for a pattern whose
/// operation is a single [`DialectOp`]: the operation's name becomes the
/// pattern's root, and the pattern receives only operations of that type.
/// A pattern that matches several operations, a whole dialect, or roots
/// known only at run time implements [`RewritePattern`] directly.
pub trait OpRewritePattern {
    /// The operation this pattern rewrites.
    type Op: DialectOp;

    /// Attempt to rewrite `op`; see [`RewritePattern::match_and_rewrite`].
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: Self::Op,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool;

    /// Optional: return a human-readable name for debugging.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

impl<P: OpRewritePattern> RewritePattern for P {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(op) = P::Op::from_op(ctx, op) else {
            return false;
        };
        OpRewritePattern::match_and_rewrite(self, ctx, op, rewriter)
    }

    fn roots(&self) -> Option<&[OpName]> {
        Some(&[P::Op::NAME])
    }

    fn name(&self) -> &'static str {
        OpRewritePattern::name(self)
    }
}
