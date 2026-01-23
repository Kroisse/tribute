//! Wasm backend that emits WebAssembly binaries from wasm.* TrunkIR operations.

mod data_registry;
pub mod lower_wasm;
pub mod passes;
pub mod translate;
pub mod type_converter;
mod validation;

pub use validation::{ValidationError, validate_wasm_ir};

pub use data_registry::{DataEntry, DataRegistry};

// Re-export emit, errors, and gc_types from trunk-ir-wasm-backend
pub use translate::{WasmBinary, compile_to_wasm};
pub use trunk_ir_wasm_backend::emit_wasm;
pub use trunk_ir_wasm_backend::gc_types;
pub use trunk_ir_wasm_backend::{CompilationError, CompilationErrorKind, CompilationResult};
