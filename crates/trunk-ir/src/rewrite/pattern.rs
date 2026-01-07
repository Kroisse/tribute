//! Rewrite pattern trait.
//!
//! Defines the interface for IR transformation patterns,
//! inspired by MLIR's RewritePattern.

use crate::Operation;

use super::op_adaptor::OpAdaptor;
use super::result::RewriteResult;

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
/// The `OpAdaptor` provides:
/// - Access to remapped operands via `adaptor.operands()`
/// - Value type lookup including block arguments via `adaptor.get_value_type()`
///
/// # Example
///
/// ```ignore
/// use trunk_ir::rewrite::{OpAdaptor, RewritePattern, RewriteResult};
///
/// struct RenamePattern;
///
/// impl RewritePattern for RenamePattern {
///     fn match_and_rewrite<'db>(
///         &self,
///         db: &'db dyn salsa::Database,
///         op: &Operation<'db>,
///         adaptor: &OpAdaptor<'db, '_>,
///     ) -> RewriteResult<'db> {
///         if op.dialect(db) != "test" || op.name(db) != "source" {
///             return RewriteResult::Unchanged;
///         }
///         // Access remapped operands and their types
///         if let Some(operand) = adaptor.operand(0) {
///             let ty = adaptor.get_value_type(db, operand);
///             // ...
///         }
///         let new_op = op.modify(db).name_str("target").build();
///         RewriteResult::Replace(new_op)
///     }
/// }
/// ```
pub trait RewritePattern {
    /// Attempt to match and rewrite an operation.
    ///
    /// Returns `RewriteResult::Unchanged` if the pattern doesn't apply.
    /// Otherwise returns the transformation result.
    ///
    /// The `adaptor` provides access to:
    /// - Remapped operands via `adaptor.operands()`
    /// - Value types (including block arguments) via `adaptor.get_value_type()`
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        adaptor: &OpAdaptor<'db, '_>,
    ) -> RewriteResult<'db>;

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
