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
//! - [`tdnr`]: Type-directed name resolution (AST → AST)
//! - [`ast_to_ir`]: AST to TrunkIR lowering
//! - [`query`]: Salsa-tracked query functions for incremental compilation
//! - [`source_file`]: Source file management and URI handling

pub mod ast;
pub mod ast_to_ir;
pub mod astgen;
pub mod query;
pub mod resolve;
pub mod source_file;
pub mod tdnr;
pub mod typeck;

pub use fluent_uri::Uri;
pub use query::{ParsedCst, parse_cst};
pub use source_file::{SourceCst, derive_module_name_from_path, path_to_uri};

use trunk_ir::{Symbol, SymbolVec, smallvec::SmallVec};

/// Build a module path for a struct field or ability operation.
///
/// Appends the type name to the module path, creating a path like ["foo", "Point"]
/// for a field accessor like `Point::x` in module `foo`.
pub fn build_field_module_path(module_path: &[Symbol], type_name: Symbol) -> SymbolVec {
    let mut path: SymbolVec = SmallVec::from_slice(module_path);
    path.push(type_name);
    path
}
