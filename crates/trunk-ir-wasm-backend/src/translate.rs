//! Language-agnostic WebAssembly module translation.
//!
//! This module provides functions for validating and emitting WebAssembly binaries
//! from TrunkIR modules that have already been lowered to the wasm dialect.

use trunk_ir::IrContext;
use trunk_ir::Module;

use crate::emit::emit_wasm;
use crate::{CompilationResult, validate_wasm_ir};

/// A compiled WebAssembly module.
pub struct WasmBinary {
    /// The compiled WebAssembly binary (bytes that can be written to .wasm file).
    pub bytes: Vec<u8>,
}

/// Emit a WebAssembly binary from a lowered TrunkIR module.
///
/// This function assumes the module has already been lowered to wasm dialect
/// and all type conversions have been resolved. It:
/// 1. Validates the IR (checks for unresolved types and non-wasm ops)
/// 2. Emits the wasm binary
pub fn emit_module_to_wasm(ctx: &mut IrContext, module: Module) -> CompilationResult<WasmBinary> {
    // Validate IR (check for unresolved types and non-wasm ops)
    validate_wasm_ir(ctx, module)?;

    // Emit wasm binary
    let bytes = emit_wasm(ctx, module)?;

    Ok(WasmBinary { bytes })
}
