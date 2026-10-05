//! WASM lowering passes for Tribute.
//!
//! This module contains Tribute-specific passes that lower high-level Tribute IR
//! to WebAssembly dialect operations.
//!
//! ## Passes
//!
//! - `convert_signatures`: Convert function and indirect-call signatures to Wasm types
//! - `tribute_rt_to_wasm`: Lower boxing/unboxing operations to wasm equivalents
//! - `const_to_wasm`: Lower string/bytes constants to wasm data segments
//! - `bytes`: Bytes layout types and the in-boundary bytes read intrinsic
//! - `struct_layouts`: Structural GC struct types of user structs and variants
//! - `intrinsic_to_wasm`: Bind the `extern "C"` bytes helpers to GC operations
//! - `wasm_gc_to_wasm`: Resolve semantic GC types to indexed WASM operations
//! - `evidence_to_wasm`: Lower evidence runtime functions to inline WASM operations
//! - `runtime_bindings`: The C runtime helpers the Wasm target binds
//! - `lower`: Main orchestrator for lowering mid-level IR to WASM
//! - `type_converter`: WASM type converter for IR-level type transformations

pub mod adt_to_wasm;
pub mod bytes;
pub mod const_to_wasm;
pub mod convert_signatures;
pub mod descriptors;
pub mod evidence_to_wasm;
pub mod intrinsic_to_wasm;
pub mod io;
pub mod lower;
pub mod runtime_bindings;
pub mod struct_layouts;
pub mod tribute_rt_to_wasm;
pub mod type_converter;
