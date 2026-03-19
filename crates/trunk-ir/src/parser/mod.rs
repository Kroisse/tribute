//! IR text format parser.
//!
//! The `raw` module provides the winnow-based stage-1 parser that produces
//! unresolved parse trees. The `builder` module consumes these to produce
//! arena IR.

pub mod builder;
pub mod raw;

pub use builder::{parse_module, parse_test_module};
pub use raw::ParseError;
