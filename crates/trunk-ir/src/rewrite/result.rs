//! Rewrite result types.
//!
//! Defines the possible outcomes of a pattern rewrite operation.

use crate::{Operation, Value};

/// Result of attempting to rewrite an operation.
///
/// This is returned by `RewritePattern::match_and_rewrite` to indicate
/// what happened during the rewrite attempt.
#[derive(Debug)]
pub enum RewriteResult<'db> {
    /// The pattern did not match or decided not to transform the operation.
    /// The original operation should be kept unchanged.
    Unchanged,

    /// Replace the operation with a single new operation.
    /// The new operation's results are mapped 1:1 to the original's results.
    Replace(Operation<'db>),

    /// Replace the operation with multiple operations.
    /// The LAST operation's results are mapped to the original's results.
    /// This supports the common pattern where earlier operations produce
    /// intermediate values, and the final operation produces the result.
    Expand(Vec<Operation<'db>>),

    /// Delete the operation and replace its results with other values.
    /// The replacement values must match the original result count.
    Erase {
        /// Values to substitute for the erased operation's results.
        replacement_values: Vec<Value<'db>>,
    },
}

impl<'db> RewriteResult<'db> {
    /// Check if this result represents a change.
    pub fn is_changed(&self) -> bool {
        !matches!(self, RewriteResult::Unchanged)
    }

    /// Create a replace result from an operation.
    pub fn replace(op: Operation<'db>) -> Self {
        RewriteResult::Replace(op)
    }

    /// Create an expand result from multiple operations.
    pub fn expand(ops: Vec<Operation<'db>>) -> Self {
        RewriteResult::Expand(ops)
    }

    /// Create an erase result with replacement values.
    pub fn erase(replacement_values: Vec<Value<'db>>) -> Self {
        RewriteResult::Erase { replacement_values }
    }
}
