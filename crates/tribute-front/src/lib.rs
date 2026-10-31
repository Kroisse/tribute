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
//! - [`ast_to_ir`]: AST to TrunkIR lowering
//! - [`query`]: Salsa-tracked query functions for incremental compilation
//! - [`sorted_map`]: Sorted-vector map for deterministic phase results
//! - [`source_file`]: Source file management and URI handling
//! - [`source_tree`]: A package's root file and file modules read into one AST
//! - [`manifest`]: Package manifests and the package graph they declare
//! - [`build_inputs`]: The files a build reads, for listing and watching them

pub mod analysis;
pub mod ast;
pub mod ast_to_ir;
pub mod astgen;
pub mod build_inputs;
mod collections;
pub mod keywords;
pub mod manifest;
pub mod monomorphize;
pub mod query;
pub mod resolve;
pub mod sorted_map;
pub mod source_file;
pub mod source_tree;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod typeck;

pub use fluent_uri::Uri;
pub use query::{ParsedCst, parse_cst};
pub use sorted_map::SortedMap;
pub use source_file::{SourceCst, derive_module_name_from_path, path_to_uri, uri_to_path};

use trunk_ir::Symbol;

/// Whether a declaration named `name` in the module whose identity path is
/// `module` is the entry point of an executable build: `main` in the root
/// module of the executable's root package.
pub(crate) fn is_root_main(name: &Symbol, module: &[Symbol]) -> bool {
    *name == "main" && module == ["pkg"]
}
