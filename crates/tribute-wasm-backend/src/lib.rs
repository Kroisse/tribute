//! Wasm backend that emits WebAssembly binaries from wasm.* TrunkIR operations.

mod emit;
mod errors;
pub mod gc_types;
pub mod lower_wasm;
pub mod passes;
mod plan;
pub mod translate;
pub mod type_converter;

pub use emit::emit_wasm;
pub use errors::{CompilationError, CompilationResult};
pub use translate::{WasmBinary, compile_to_wasm};

// Shared attribute symbols used across modules
trunk_ir::symbols! {
    ATTR_SYM_NAME => "sym_name",
    ATTR_MODULE => "module",
    ATTR_NAME => "name",
}
