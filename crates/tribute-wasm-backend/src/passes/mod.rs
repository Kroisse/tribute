//! Lowering passes from mid-level IR to wasm dialect.
//!
//! Each pass converts a specific dialect to wasm operations.
//!
//! Language-agnostic passes (arith_to_wasm, func_to_wasm, scf_to_wasm, adt_to_wasm,
//! trampoline_to_wasm) are provided by trunk-ir-wasm-backend and used directly
//! in lower_wasm.rs.

pub mod const_to_wasm;
pub mod intrinsic_to_wasm;
pub mod normalize_primitive_types;
pub mod tribute_rt_to_wasm;
pub mod wasm_gc_type_assign;
pub mod wasm_type_concrete;
