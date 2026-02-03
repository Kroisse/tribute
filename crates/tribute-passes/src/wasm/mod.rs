//! WASM lowering passes for Tribute.
//!
//! This module contains Tribute-specific passes that lower high-level Tribute IR
//! to WebAssembly dialect operations.
//!
//! ## Passes
//!
//! - `normalize_primitive_types`: Normalize tribute_rt types to core/wasm types
//! - `tribute_rt_to_wasm`: Lower boxing/unboxing operations to wasm equivalents
//! - `const_to_wasm`: Lower string/bytes constants to wasm data segments
//! - `intrinsic_to_wasm`: Lower intrinsic calls to WASM operations
//! - `wasm_gc_type_assign`: Assign unique type indices to GC struct types
//! - `evidence_to_wasm`: Lower evidence runtime functions to inline WASM operations
//! - `handler_table_to_wasm`: Lower ability.handler_table to wasm.table + wasm.elem
//! - `lower`: Main orchestrator for lowering mid-level IR to WASM
//! - `type_converter`: WASM type converter for IR-level type transformations

pub mod const_to_wasm;
pub mod evidence_to_wasm;
pub mod handler_table_to_wasm;
pub mod intrinsic_to_wasm;
pub mod lower;
pub mod normalize_primitive_types;
pub mod tribute_rt_to_wasm;
pub mod type_converter;
pub mod wasm_gc_type_assign;
