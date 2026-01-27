//! LSP-specific indexes for AST-based lookups.
//!
//! This module provides Salsa-tracked indexes for LSP features like hover,
//! go-to-definition, and find-references. These indexes are built from the
//! typed AST (`Module<TypedRef>`) rather than TrunkIR, enabling incremental
//! updates based on AST changes.
//!
//! ## Indexes
//!
//! - **Type Index**: Maps source positions to inferred types (for hover)
//! - **Definition Index**: Maps names to definitions and references (for go-to-definition)
//! - **Function Signatures**: Collects function signature info (for signature help)
//! - **Completion Index**: Provides completion candidates
//! - **Document Symbols**: Provides document outline

mod completion;
mod definition_index;
mod type_index;

pub use completion::*;
pub use definition_index::*;
pub use type_index::*;
