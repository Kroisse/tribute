//! Language-agnostic WebAssembly module translation.
//!
//! This module provides functions for validating and emitting WebAssembly binaries
//! from TrunkIR modules that have already been lowered to the wasm dialect.

use trunk_ir::IrContext;
use trunk_ir::Module;
use trunk_ir::Symbol;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;

use crate::{CompilationResult, emit_wasm, validate_wasm_ir};

/// A compiled WebAssembly module with metadata.
pub struct WasmBinary {
    /// The compiled WebAssembly binary (bytes that can be written to .wasm file).
    pub bytes: Vec<u8>,
    /// Exported function names from this module.
    pub exports: Vec<Symbol>,
    /// Imported functions: (module_name, function_name).
    pub imports: Vec<(Symbol, Symbol)>,
}

/// Emit a WebAssembly binary from a lowered TrunkIR module (arena version).
///
/// This function assumes the module has already been lowered to wasm dialect
/// and all type conversions have been resolved. It:
/// 1. Validates the IR (checks for unresolved types and non-wasm ops)
/// 2. Emits the wasm binary
/// 3. Extracts metadata (exports, imports)
pub fn emit_module_to_wasm(ctx: &mut IrContext, module: Module) -> CompilationResult<WasmBinary> {
    // Validate IR (check for unresolved types and non-wasm ops)
    validate_wasm_ir(ctx, module)?;

    // Emit wasm binary
    let bytes = emit_wasm(ctx, module)?;

    // Extract exports and imports from module
    let (exports, imports) = extract_metadata(ctx, module);

    Ok(WasmBinary {
        bytes,
        exports,
        imports,
    })
}

/// Extract metadata (exports and imports) from a compiled module.
fn extract_metadata(ctx: &IrContext, module: Module) -> (Vec<Symbol>, Vec<(Symbol, Symbol)>) {
    let mut exports = Vec::new();
    let mut imports = Vec::new();

    let Some(body) = module.body(ctx) else {
        return (exports, imports);
    };

    for &block_ref in &ctx.region(body).blocks {
        for &op in &ctx.block(block_ref).ops {
            if let Ok(export_op) = wasm_dialect::ExportFunc::from_op(ctx, op) {
                let name = export_op.name(ctx);
                exports.push(Symbol::from_dynamic(&name));
            } else if let Ok(import_op) = wasm_dialect::ImportFunc::from_op(ctx, op) {
                let module_name = import_op.module(ctx);
                let func_name = import_op.name(ctx);
                imports.push((module_name, func_name));
            }
        }
    }

    (exports, imports)
}
