//! WASM type converter for IR-level type transformations.
//!
//! This module re-exports the type converter from tribute-passes.
//! The implementation has been moved to tribute-passes as part of
//! the tribute-wasm-backend refactoring (issue #161).

// Re-export from tribute-passes
pub use tribute_passes::wasm_type_converter::{closure_adt_type, wasm_type_converter};
