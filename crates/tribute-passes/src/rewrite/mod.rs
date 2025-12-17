//! IR rewriting infrastructure for TrunkIR.
//!
//! This module provides a pattern-based rewriting system inspired by MLIR's
//! `PatternRewriter`. It handles the complexities of transforming immutable
//! Salsa-tracked IR structures.
//!
//! # Overview
//!
//! The rewriter operates on TrunkIR operations through three key components:
//!
//! - [`RewritePattern`]: Trait for defining transformation patterns
//! - [`RewriteContext`]: Tracks value mappings during rewrites
//! - [`PatternApplicator`]: Drives pattern application to fixpoint
//!
//! # Usage
//!
//! ```ignore
//! // Define a pattern
//! struct MyPattern;
//!
//! impl<'db> RewritePattern<'db> for MyPattern {
//!     fn match_and_rewrite(
//!         &self,
//!         db: &'db dyn Database,
//!         op: &Operation<'db>,
//!         ctx: &mut RewriteContext<'db>,
//!     ) -> RewriteResult<'db> {
//!         if !op.matches(db, "src", "var") {
//!             return RewriteResult::Unchanged;
//!         }
//!         // ... transform the operation ...
//!         RewriteResult::Replace(new_op)
//!     }
//! }
//!
//! // Apply patterns to a module
//! let applicator = PatternApplicator::new()
//!     .add_pattern(MyPattern)
//!     .with_max_iterations(50);
//!
//! let result = applicator.apply(db, module);
//! assert!(result.reached_fixpoint);
//! ```
//!
//! # Design Notes
//!
//! Since TrunkIR uses Salsa's tracked structures (which are immutable),
//! the rewriter uses a functional style that rebuilds the IR tree.
//! The `RewriteContext` maintains value mappings so that when an
//! operation is replaced, subsequent operations can reference the
//! new values.

mod applicator;
mod context;
mod pattern;
mod result;

pub use applicator::{ApplyResult, PatternApplicator};
pub use context::RewriteContext;
pub use pattern::{OperationMatcher, RewritePattern};
pub use result::RewriteResult;
