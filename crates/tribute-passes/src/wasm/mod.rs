//! WASM lowering passes for Tribute.
//!
//! This module contains Tribute-specific passes that lower high-level Tribute IR
//! to WebAssembly dialect operations.
//!
//! ## Passes
//!
//! - `normalize_primitive_types`: Normalize tribute_rt types to core/wasm types
//! - `tribute_rt_to_wasm`: Lower boxing/unboxing operations to wasm equivalents
//! - `wasm_type_concrete`: Concretize type variables in wasm operations
//! - `const_to_wasm`: Lower string/bytes constants to wasm data segments
//! - `intrinsic_to_wasm`: Lower intrinsic calls to WASM operations
//! - `wasm_gc_type_assign`: Assign unique type indices to GC struct types

pub mod const_to_wasm;
pub mod intrinsic_to_wasm;
pub mod normalize_primitive_types;
pub mod tribute_rt_to_wasm;
pub mod wasm_gc_type_assign;
pub mod wasm_type_concrete;
