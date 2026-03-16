//! TrunkIR to WASM lowering passes.
//!
//! These passes lower language-agnostic TrunkIR dialects to WASM operations.

pub mod adt_to_wasm;
pub mod arith_to_wasm;
pub mod func_to_wasm;
pub mod scf_to_wasm;
