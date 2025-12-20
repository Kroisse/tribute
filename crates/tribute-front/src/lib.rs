//! Front-end utilities for Tribute.
//!
//! This crate provides Tree-sitter CST parsing and lowering utilities,
//! along with shared text helpers for editor integrations.

pub mod source_file;
pub mod tirgen;

pub use fluent_uri::Uri;
pub use source_file::{SourceCst, SourceFile, path_to_uri, uri_to_path};
pub use tirgen::{
    ParsedCst, lower_cst, lower_source_cst, lower_source_file, parse_cst, parse_cst_from_tree,
};
