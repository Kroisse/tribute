//! Language-agnostic WebAssembly module translation.
//!
//! This module provides functions for validating and emitting WebAssembly binaries
//! from TrunkIR modules that have already been lowered to the wasm dialect.

use trunk_ir::IrContext;
use trunk_ir::Module;

use trunk_ir::analysis::AnalysisCache;
use trunk_ir::dialect::{core, wasm};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::PassManager;

use crate::emit::emit_wasm;
use crate::passes::explicit_locals::ExplicitLocals;
use crate::validation::validate_wasm_stack;
use crate::{CompilationError, CompilationResult, validate_wasm_ir};

/// Run the emission passes that make every local access explicit.
pub(crate) fn assign_explicit_locals(ctx: &mut IrContext, module: Module) -> CompilationResult<()> {
    let root = core::Module::from_op(ctx, module.op())
        .map_err(|_| CompilationError::invalid_module("Wasm emission runs on a core.module"))?;
    let mut pm = PassManager::new();
    pm.with_debug_verifier();
    pm.nest::<wasm::Func>().add_pass(ExplicitLocals);
    pm.run(ctx, root, &mut AnalysisCache::new())
        .map_err(|error| CompilationError::invalid_module(error.to_string()))
}

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
/// 2. Assigns the locals with `ExplicitLocals` and checks the stack contract
/// 3. Emits the wasm binary
pub fn emit_module_to_wasm(ctx: &mut IrContext, module: Module) -> CompilationResult<WasmBinary> {
    // Validate IR (check for unresolved types and non-wasm ops)
    validate_wasm_ir(ctx, module)?;

    assign_explicit_locals(ctx, module)?;
    validate_wasm_stack(ctx, module)?;

    // Emit wasm binary
    let bytes = emit_wasm(ctx, module)?;

    Ok(WasmBinary { bytes })
}
