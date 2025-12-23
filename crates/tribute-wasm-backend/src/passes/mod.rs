//! Lowering passes from mid-level IR to wasm dialect.
//!
//! Each pass converts a specific dialect to wasm operations.

pub mod adt_to_wasm;
pub mod arith_to_wasm;
pub mod func_to_wasm;
pub mod scf_to_wasm;
