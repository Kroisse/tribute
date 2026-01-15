//! Lowering passes from mid-level IR to wasm dialect.
//!
//! Each pass converts a specific dialect to wasm operations.

pub mod adt_to_wasm;
pub mod arith_to_wasm;
pub mod const_to_wasm;
pub mod func_to_wasm;
pub mod intrinsic_to_wasm;
pub mod scf_to_wasm;
pub mod trampoline_to_wasm;
pub mod tribute_rt_to_wasm;
pub mod wasm_gc_type_assign;
pub mod wasm_type_concrete;
