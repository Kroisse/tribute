//! Lower mid-level IR to WebAssembly dialect operations.
//!
//! This module re-exports the lowering implementation from tribute-passes.
//! The implementation has been moved to tribute-passes as part of
//! the tribute-wasm-backend refactoring (issue #161).

// Re-export from tribute-passes
pub use tribute_passes::wasm::lower::lower_to_wasm;
