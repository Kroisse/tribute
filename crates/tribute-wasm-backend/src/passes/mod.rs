//! Lowering passes from mid-level IR to wasm dialect.
//!
//! Each pass converts a specific dialect to wasm operations.
//!
//! Language-agnostic passes (arith_to_wasm, func_to_wasm, scf_to_wasm, adt_to_wasm,
//! trampoline_to_wasm) are provided by trunk-ir-wasm-backend and used directly
//! in lower_wasm.rs.
//!
//! Tribute-specific passes are now implemented in tribute-passes and re-exported here.

pub use tribute_passes::wasm::const_to_wasm;
pub use tribute_passes::wasm::intrinsic_to_wasm;
pub use tribute_passes::wasm::normalize_primitive_types;
pub use tribute_passes::wasm::tribute_rt_to_wasm;
pub use tribute_passes::wasm::wasm_gc_type_assign;
pub use tribute_passes::wasm::wasm_type_concrete;
