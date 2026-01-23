//! Compiled WebAssembly binary representation.
//!
//! This module provides the `WasmBinary` type which represents a fully compiled
//! WebAssembly artifact with binary code and metadata.

use trunk_ir::Symbol;

/// A compiled WebAssembly module with metadata.
///
/// This is a Salsa tracked struct that represents a fully compiled WebAssembly
/// artifact with binary code and metadata for tooling integration.
#[salsa::tracked]
pub struct WasmBinary<'db> {
    /// The compiled WebAssembly binary (bytes that can be written to .wasm file).
    #[returns(ref)]
    pub bytes: Vec<u8>,

    /// Exported function names from this module.
    #[returns(ref)]
    pub exports: Vec<Symbol>,

    /// Imported functions: (module_name, function_name).
    #[returns(ref)]
    pub imports: Vec<(Symbol, Symbol)>,
}
