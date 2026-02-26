//! Arena rewrite pattern trait.
//!
//! Defines the interface for arena IR transformation patterns.
//! No lifetimes needed — arena refs are Copy.

use super::rewriter::PatternRewriter;
use crate::arena::context::IrContext;
use crate::arena::refs::OpRef;

/// A pattern that can match and transform arena IR operations.
///
/// This is the arena equivalent of `RewritePattern<'db>`. Since arena IR
/// uses `OpRef` (Copy, no lifetime), the trait itself has no lifetime parameter.
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
pub trait ArenaRewritePattern {
    /// Attempt to match and rewrite an operation.
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter,
    ) -> bool;

    /// Optional: return a human-readable name for debugging.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}
