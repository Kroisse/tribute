//! Rewrite pattern trait.
//!
//! Defines the interface for IR transformation patterns,
//! inspired by MLIR's RewritePattern.

use tribute_trunk_ir::Operation;

use super::context::RewriteContext;
use super::result::RewriteResult;

/// A pattern that can match and transform IR operations.
///
/// This is the core abstraction for IR transformation. Each pattern
/// implements `match_and_rewrite` which both checks if the pattern
/// applies and performs the transformation in one step.
///
/// # Example
///
/// ```ignore
/// struct ResolveSrcVar {
///     env: ModuleEnv,
/// }
///
/// impl RewritePattern for ResolveSrcVar {
///     fn match_and_rewrite<'db>(
///         &self,
///         db: &'db dyn Database,
///         op: &Operation<'db>,
///         ctx: &mut RewriteContext<'db>,
///     ) -> RewriteResult<'db> {
///         // Check if this is a src.var operation
///         if op.dialect(db).text(db) != "src" || op.name(db).text(db) != "var" {
///             return RewriteResult::Unchanged;
///         }
///
///         // Perform the transformation...
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
    /// The context can be used to look up mapped values for operands
    /// and to register new value mappings for results.
    fn match_and_rewrite<'db>(
        &self,
        db: &'db dyn salsa::Database,
        op: &Operation<'db>,
        ctx: &mut RewriteContext<'db>,
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
        self.dialect(db).text(db) == dialect && self.name(db).text(db) == name
    }

    fn is_dialect(&self, db: &dyn salsa::Database, dialect: &str) -> bool {
        self.dialect(db).text(db) == dialect
    }
}
