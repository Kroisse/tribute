//! WASM backend for TrunkIR.
//!
//! This crate provides language-agnostic lowering passes and emission for converting
//! TrunkIR dialects to WebAssembly binaries.
//!
//! ## Passes
//!
//! - `func_to_wasm`: Lowers `func.*` operations to `wasm.*`
//! - `arith_to_wasm`: Lowers `arith.*` operations to `wasm.*`
//! - `scf_to_wasm`: Lowers `scf.*` operations to `wasm.*`
//!
//! ## Emission
//!
//! - `emit`: Emits WebAssembly binaries from `wasm.*` TrunkIR operations
//! - `gc_types`: GC type registry for WebAssembly GC types
//! - `translate`: Language-agnostic module translation (validate + emit)
//! - `data_registry`: Registry for WASM data section entries
//! - `plan`: Memory and export planning metadata

mod data_registry;
mod emit;
mod errors;
pub mod gc_types;
pub mod passes;
mod plan;
mod translate;
mod validation;
mod wasm_binary;

pub use data_registry::{DataEntry, DataRegistry};
pub use emit::emit_wasm;
pub use errors::{CompilationError, CompilationErrorKind, CompilationResult};
pub use plan::{MainExports, MemoryPlan};
pub use translate::emit_module_to_wasm;
pub use validation::{ValidationError, validate_wasm_ir};
pub use wasm_binary::WasmBinary;
