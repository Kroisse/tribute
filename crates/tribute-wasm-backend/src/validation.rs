//! IR validation for wasm backend.
//!
//! This module re-exports the validation from trunk-ir-wasm-backend.
//! The implementation has been moved to trunk-ir-wasm-backend as part of
//! the tribute-wasm-backend refactoring (issue #161).

// Re-export from trunk-ir-wasm-backend
pub use trunk_ir_wasm_backend::{ValidationError, validate_wasm_ir};
