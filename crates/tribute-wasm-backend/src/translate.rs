//! Compilation from TrunkIR to WebAssembly binary.
//!
//! This module provides the Tribute-specific entry point for compiling a TrunkIR module
//! to a WebAssembly binary. It combines Tribute-specific lowering with language-agnostic
//! emission from trunk-ir-wasm-backend.

use crate::type_converter::wasm_type_converter;
use trunk_ir::conversion::resolve_unrealized_casts;
use trunk_ir::dialect::core::Module;

// Re-export for convenience
pub use crate::CompilationResult;

// Re-export WasmBinary from trunk-ir-wasm-backend
pub use trunk_ir_wasm_backend::WasmBinary;

/// Compile a TrunkIR module to WebAssembly binary.
///
/// This is a Salsa tracked function that:
/// 1. Lowers the module from func/scf/arith dialects to wasm dialect operations
/// 2. Resolves unrealized conversion casts using Tribute-specific type converter
/// 3. Validates and emits the wasm binary (delegated to trunk-ir-wasm-backend)
///
/// Note: Dead code elimination (DCE) is performed earlier in the pipeline
/// (stage_dce) before this function is called, ensuring unused functions
/// are already removed before lowering.
///
/// Memoization is automatic via Salsa's incremental compilation system.
#[salsa::tracked]
pub fn compile_to_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> crate::CompilationResult<WasmBinary<'db>> {
    // Phase 1 - Lower to wasm dialect (Tribute-specific)
    let lowered = crate::lower_wasm::lower_to_wasm(db, module);

    // Phase 2 - Resolve unrealized_conversion_cast operations (Tribute-specific type converter)
    let type_converter = wasm_type_converter();
    let lowered = resolve_unrealized_casts(db, lowered, &type_converter)
        .map(|r| r.module)
        .map_err(crate::CompilationError::unresolved_casts)?;

    // Phase 3 - Validate and emit (language-agnostic, delegated to trunk-ir-wasm-backend)
    trunk_ir_wasm_backend::emit_module_to_wasm(db, lowered)
}
