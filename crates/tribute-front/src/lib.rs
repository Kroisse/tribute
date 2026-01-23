//! Front-end utilities for Tribute.
//!
//! This crate provides Tree-sitter CST parsing and lowering utilities,
//! along with shared text helpers for editor integrations.
//!
//! ## Modules
//!
//! - [`ast`]: Salsa-tracked AST types with phase-parameterized name resolution
//! - [`source_file`]: Source file management and URI handling
//! - [`tirgen`]: CST to TrunkIR lowering (legacy, to be replaced by astgen)

pub mod ast;
pub mod source_file;
pub mod tirgen;

pub use fluent_uri::Uri;
pub use source_file::{SourceCst, path_to_uri};
pub use tirgen::{ParsedCst, lower_cst, lower_source_cst, parse_cst};
