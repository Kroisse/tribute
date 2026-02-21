//! Rewrite pattern trait.
//!
//! Defines the interface for IR transformation patterns,
//! inspired by MLIR's RewritePattern.

use crate::Operation;

use super::rewriter::PatternRewriter;

/// A pattern that can match and transform IR operations.
///
/// This is the core abstraction for IR transformation. Each pattern
/// implements `match_and_rewrite` which both checks if the pattern
/// applies and performs the transformation in one step.
///
/// The `PatternApplicator` handles all value remapping automatically:
/// - Operands are remapped before calling `match_and_rewrite`
/// - Results are mapped after the pattern returns
///
/// # Arguments
///
/// - `op`: The **original** operation. Use this for dialect/name matching
///   via `from_operation(db, *op)`, and for accessing attributes and regions.
/// - `rewriter`: The `PatternRewriter` providing:
///   - Remapped operands via `rewriter.operand(i)` / `rewriter.operands()`
///   - Value type lookup via `rewriter.get_value_type()`
///   - Mutation methods: `replace_op()`, `insert_op()`, `erase_op()`, `add_module_op()`
///
/// **Important**: Do NOT use `op.operands(db)` — those are the original
/// (possibly stale) operands. Always use `rewriter.operand(i)` instead.
///
/// # Return Value
///
/// Return `true` if the pattern matched and mutations were recorded.
/// Return `false` if the pattern does not apply.
///
/// # Example
///
/// ```no_run
/// use trunk_ir::Operation;
/// use trunk_ir::rewrite::{PatternRewriter, RewritePattern};
///
/// struct RenamePattern;
///
/// impl<'db> RewritePattern<'db> for RenamePattern {
///     fn match_and_rewrite(
///         &self,
///         db: &'db dyn salsa::Database,
///         op: &Operation<'db>,
///         rewriter: &mut PatternRewriter<'db, '_>,
///     ) -> bool {
///         if op.dialect(db) != "test" || op.name(db) != "source" {
///             return false;
///         }
///         // Access remapped operands and their types
///         if let Some(operand) = rewriter.operand(0) {
///             let _ty = rewriter.get_value_type(db, operand);
///         }
///         let new_op = op.modify(db).name_str("target").build();
///         rewriter.replace_op(new_op);
///         true
///     }
/// }
/// ```
pub trait RewritePattern<'db> {
    /// Attempt to match and rewrite an operation.
    ///
    /// Returns `false` if the pattern doesn't apply.
    /// Returns `true` if the pattern matched and mutations were recorded
    /// on the `rewriter`.
    ///
    /// The `op` parameter is the **original** operation (for matching).
    /// Use `rewriter.operand(i)` for remapped operands, not `op.operands(db)`.
    fn match_and_rewrite(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        rewriter: &mut PatternRewriter<'db, '_>,
    ) -> bool;

    /// Optional: return a human-readable name for debugging.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Helper trait for dialect/operation name matching.
pub trait OperationMatcher {
    /// Check if an operation matches a specific dialect and name.
    fn matches(&self, db: &dyn salsa::Database, dialect: &str, name: &str) -> bool;

    /// Check if an operation is from a specific dialect.
    fn is_dialect(&self, db: &dyn salsa::Database, dialect: &str) -> bool;
}

impl<'db> OperationMatcher for Operation<'db> {
    fn matches(&self, db: &dyn salsa::Database, dialect: &str, name: &str) -> bool {
        self.dialect(db) == dialect && self.name(db) == name
    }

    fn is_dialect(&self, db: &dyn salsa::Database, dialect: &str) -> bool {
        self.dialect(db) == dialect
    }
}
