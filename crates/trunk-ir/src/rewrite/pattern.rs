//! Rewrite pattern trait.
//!
//! Defines the interface for IR transformation patterns,
//! inspired by MLIR's RewritePattern.

use crate::Operation;

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
/// ```
/// # use salsa::Database;
/// # use trunk_ir::test_db::TestDatabase;
/// # use trunk_ir::{Location, Operation, PathId, Span};
/// use trunk_ir::rewrite::{RewriteContext, RewritePattern, RewriteResult};
///
/// struct RenamePattern;
///
/// impl RewritePattern for RenamePattern {
///     fn match_and_rewrite<'db>(
///         &self,
///         db: &'db dyn salsa::Database,
///         op: &Operation<'db>,
///         _ctx: &mut RewriteContext<'db>,
///     ) -> RewriteResult<'db> {
///         if op.dialect(db) != "test" || op.name(db) != "source" {
///             return RewriteResult::Unchanged;
///         }
///         let new_op = op.modify(db).name_str("target").build();
///         RewriteResult::Replace(new_op)
///     }
/// }
/// # #[salsa::tracked]
/// # fn make_op(db: &dyn salsa::Database) -> Operation<'_> {
/// #     let path = PathId::new(db, "file:///test.trb".to_owned());
/// #     let location = Location::new(path, Span::new(0, 0));
/// #     Operation::of_name(db, location, "test.source").build()
/// # }
/// # #[salsa::tracked]
/// # fn rewrite_once(db: &dyn salsa::Database, op: Operation<'_>) -> String {
/// #     let mut ctx = RewriteContext::new();
/// #     let result = RenamePattern.match_and_rewrite(db, &op, &mut ctx);
/// #     match result {
/// #         RewriteResult::Replace(new_op) => new_op.full_name(db),
/// #         _ => "unchanged".to_string(),
/// #     }
/// # }
/// # TestDatabase::default().attach(|db| {
/// #     let op = make_op(db);
/// #     let result = rewrite_once(db, op);
/// #     assert_eq!(result, "test.target");
/// # });
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
        self.dialect(db) == dialect && self.name(db) == name
    }

    fn is_dialect(&self, db: &dyn salsa::Database, dialect: &str) -> bool {
        self.dialect(db) == dialect
    }
}
