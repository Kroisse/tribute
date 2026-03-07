//! IR text format parser.
//!
//! The `raw` module provides the winnow-based stage-1 parser that produces
//! unresolved parse trees. The arena IR builder in `arena/parser.rs` consumes
//! these to produce arena IR.

pub(crate) mod raw;

pub use raw::ParseError;
