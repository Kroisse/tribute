//! WASM backend for TrunkIR.
//!
//! This crate provides language-agnostic lowering passes for converting
//! TrunkIR dialects to WASM target operations.
//!
//! ## Passes
//!
//! - `func_to_wasm`: Lowers `func.*` operations to `wasm.*`
//! - `arith_to_wasm`: Lowers `arith.*` operations to `wasm.*`
//! - `scf_to_wasm`: Lowers `scf.*` operations to `wasm.*`
//!
//! Language-specific passes (like tribute_rt_to_wasm, trampoline_to_wasm)
//! remain in the language-specific backend crates.

pub mod passes;
