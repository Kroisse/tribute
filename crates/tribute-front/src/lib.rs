//! Front-end utilities for Tribute.
//!
//! This crate provides Tree-sitter CST parsing and lowering utilities,
//! along with shared text helpers for editor integrations.

pub mod line_index;
pub mod tirgen;

pub use line_index::LineIndex;
pub use tirgen::{ParsedCst, lower_cst, lower_source_file, parse_cst};
