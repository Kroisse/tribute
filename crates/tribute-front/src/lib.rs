//! Front-end utilities for Tribute.
//!
//! This crate provides Tree-sitter CST parsing and lowering utilities,
//! along with shared text helpers for editor integrations.
//!
//! ## Modules
//!
//! - [`ast`]: Salsa-tracked AST types with phase-parameterized name resolution
//! - [`astgen`]: CST to AST lowering
//! - [`resolve`]: Name resolution (AST → AST)
//! - [`typeck`]: Type checking (AST → AST)
//! - [`source_file`]: Source file management and URI handling
//! - [`tirgen`]: CST to TrunkIR lowering (legacy, to be replaced by astgen)

pub mod ast;
pub mod ast_to_ir;
pub mod astgen;
pub mod resolve;
pub mod source_file;
pub mod tdnr;
pub mod tirgen;
pub mod typeck;

pub use fluent_uri::Uri;
pub use source_file::{SourceCst, path_to_uri};
pub use tirgen::{ParsedCst, lower_cst, lower_source_cst, parse_cst};
