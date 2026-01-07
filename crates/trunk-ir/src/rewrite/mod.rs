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
//! - [`RewriteContext`]: Tracks value mappings during rewrites (internal to applicator)
//! - [`PatternApplicator`]: Drives pattern application to fixpoint
//!
//! # Usage
//!
//! ```
//! # use salsa::Database;
//! # use salsa::DatabaseImpl;
//! # use trunk_ir::{Block, BlockId, Location, Operation, PathId, Region, Span, Symbol, idvec};
//! # use trunk_ir::dialect::core::Module;
//! use trunk_ir::rewrite::{OpAdaptor, PatternApplicator, RewritePattern, RewriteResult};
//!
//! struct RenamePattern;
//!
//! impl RewritePattern for RenamePattern {
//!     fn match_and_rewrite<'db>(
//!         &self,
//!         db: &'db dyn salsa::Database,
//!         op: &Operation<'db>,
//!         _adaptor: &OpAdaptor<'db, '_>,
//!     ) -> RewriteResult<'db> {
//!         // Note: op.operands() are already remapped by the applicator
//!         if op.dialect(db) != "test" || op.name(db) != "source" {
//!             return RewriteResult::Unchanged;
//!         }
//!         let new_op = op.modify(db).name_str("target").build();
//!         RewriteResult::Replace(new_op)
//!     }
//! }
//! # #[salsa::tracked]
//! # fn make_module(db: &dyn salsa::Database) -> Module<'_> {
//! #     let path = PathId::new(db, "file:///test.trb".to_owned());
//! #     let location = Location::new(path, Span::new(0, 0));
//! #     let op = Operation::of_name(db, location, "test.source").build();
//! #     let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![op]);
//! #     let region = Region::new(db, location, idvec![block]);
//! #     Module::create(db, location, Symbol::new("test"), region)
//! # }
//! # #[salsa::tracked]
//! # fn apply_rename(db: &dyn salsa::Database, module: Module<'_>) -> bool {
//! #     let applicator = PatternApplicator::new()
//! #         .add_pattern(RenamePattern)
//! #         .with_max_iterations(50);
//! #     let result = applicator.apply(db, module);
//! #     result.reached_fixpoint
//! # }
//! # DatabaseImpl::default().attach(|db| {
//! #     let module = make_module(db);
//! let reached = apply_rename(db, module);
//! assert!(reached);
//! # });
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
mod op_adaptor;
mod pattern;
mod result;
mod type_converter;

pub use applicator::{ApplyResult, PatternApplicator};
pub use context::RewriteContext;
pub use op_adaptor::OpAdaptor;
pub use pattern::{OperationMatcher, RewritePattern};
pub use result::RewriteResult;
pub use type_converter::{MaterializeResult, OpVec, TypeConverter};
