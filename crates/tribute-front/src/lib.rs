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
pub mod monomorphize;
pub mod query;
pub mod resolve;
pub mod source_file;
pub mod tdnr;
pub mod typeck;

pub use fluent_uri::Uri;
pub use query::{ParsedCst, parse_cst};
pub use source_file::{SourceCst, derive_module_name_from_path, path_to_uri};

use trunk_ir::Symbol;

/// Build a qualified symbol by joining `prefix` and `name` with `::`.
///
/// If `prefix` is empty, returns `name` directly (no allocation).
/// Otherwise, temporarily appends `::name` to the buffer, creates the symbol,
/// then restores the buffer to its original length.
pub fn qualified_symbol(prefix: &mut String, name: Symbol) -> Symbol {
    if prefix.is_empty() {
        name
    } else {
        let len = prefix.len();
        prefix.push_str("::");
        name.with_str(|s| prefix.push_str(s));
        let sym = Symbol::from_dynamic(prefix);
        prefix.truncate(len);
        sym
    }
}

/// Push a segment onto a prefix buffer. Returns the length before push (for truncate).
pub fn push_prefix(prefix: &mut String, name: Symbol) -> usize {
    let len = prefix.len();
    if !prefix.is_empty() {
        prefix.push_str("::");
    }
    name.with_str(|s| prefix.push_str(s));
    len
}
