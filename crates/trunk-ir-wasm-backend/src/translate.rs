//! Language-agnostic WebAssembly module translation.
//!
//! This module provides functions for validating and emitting WebAssembly binaries
//! from TrunkIR modules that have already been lowered to the wasm dialect.

use trunk_ir::Symbol;
use trunk_ir::arena::ArenaModule;
use trunk_ir::arena::IrContext;
use trunk_ir::arena::dialect::wasm as arena_wasm;
use trunk_ir::arena::ops::ArenaDialectOp;

use crate::{CompilationResult, emit_wasm, validate_wasm_ir};

/// Output of arena-based WASM emission.
pub struct WasmBinaryOutput {
    pub bytes: Vec<u8>,
    pub exports: Vec<Symbol>,
    pub imports: Vec<(Symbol, Symbol)>,
}

/// Emit a WebAssembly binary from a lowered TrunkIR module (arena version).
///
/// This function assumes the module has already been lowered to wasm dialect
/// and all type conversions have been resolved. It:
/// 1. Validates the IR (checks for unresolved types and non-wasm ops)
/// 2. Emits the wasm binary
/// 3. Extracts metadata (exports, imports)
pub fn emit_module_to_wasm_arena(
    ctx: &mut IrContext,
    module: ArenaModule,
) -> CompilationResult<WasmBinaryOutput> {
    // Validate IR (check for unresolved types and non-wasm ops)
    validate_wasm_ir(ctx, module)?;

    // Emit wasm binary
    let bytes = emit_wasm(ctx, module)?;

    // Extract exports and imports from module
    let (exports, imports) = extract_metadata(ctx, module);

    Ok(WasmBinaryOutput {
        bytes,
        exports,
        imports,
    })
}

/// Extract metadata (exports and imports) from a compiled module.
fn extract_metadata(ctx: &IrContext, module: ArenaModule) -> (Vec<Symbol>, Vec<(Symbol, Symbol)>) {
    let mut exports = Vec::new();
    let mut imports = Vec::new();

    let Some(body) = module.body(ctx) else {
        return (exports, imports);
    };

    for &block_ref in &ctx.region(body).blocks {
        for &op in &ctx.block(block_ref).ops {
            if let Ok(export_op) = arena_wasm::ExportFunc::from_op(ctx, op) {
                let name = export_op.name(ctx);
                exports.push(Symbol::from_dynamic(&name));
            } else if let Ok(import_op) = arena_wasm::ImportFunc::from_op(ctx, op) {
                let module_name = import_op.module(ctx);
                let func_name = import_op.name(ctx);
                imports.push((module_name, func_name));
            }
        }
    }

    (exports, imports)
}
