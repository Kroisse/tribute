//! Compilation from TrunkIR to WebAssembly binary.
//!
//! This module provides the main entry point for compiling a Tribute TrunkIR module
//! to a WebAssembly binary. It combines lowering (IR transformation) and emission
//! (binary generation) into a single tracked artifact.

use trunk_ir::Symbol;
use trunk_ir::dialect::core::Module;

// Re-export for convenience
pub use crate::CompilationResult;

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

/// Compile a TrunkIR module to WebAssembly binary.
///
/// This function:
/// 1. Lowers the module from func/scf/arith dialects to wasm dialect operations
/// 2. Emits the wasm dialect to a WebAssembly binary
/// 3. Collects metadata (exports, imports) for tooling integration
///
/// Note: This is not a Salsa tracked function because Module is a tracked struct
/// and cannot be used as a tracked function parameter. Memoization should be
/// handled at the caller level (e.g., in the pipeline stage).
#[salsa::tracked]
pub fn compile_to_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> crate::CompilationResult<WasmBinary<'db>> {
    // TODO: Phase 1 - Lower to wasm dialect
    // For now, we'll use the existing emit_wasm function directly
    // Later we'll insert lowering before emission

    let bytes = crate::emit_wasm(db, module)?;

    // TODO: Extract exports and imports from module
    let exports = Vec::new();
    let imports = Vec::new();

    Ok(WasmBinary::new(db, bytes, exports, imports))
}
