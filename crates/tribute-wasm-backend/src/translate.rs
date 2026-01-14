//! Compilation from TrunkIR to WebAssembly binary.
//!
//! This module provides the main entry point for compiling a Tribute TrunkIR module
//! to a WebAssembly binary. It combines lowering (IR transformation) and emission
//! (binary generation) into a single tracked artifact.

use crate::{ATTR_MODULE, ATTR_NAME, ATTR_SYM_NAME};
use trunk_ir::Operation;
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
/// This is a Salsa tracked function that:
/// 1. Lowers the module from func/scf/arith dialects to wasm dialect operations
/// 2. Validates that all types are resolved (no placeholder types)
/// 3. Emits the wasm dialect to a WebAssembly binary
/// 4. Collects metadata (exports, imports) for tooling integration
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
    // Phase 1 - Lower to wasm dialect
    let lowered = crate::lower_wasm::lower_to_wasm(db, module);

    // Phase 2 - Validate IR (check for unresolved types and non-wasm ops)
    crate::validate_wasm_ir(db, lowered)?;

    // Phase 3 - Emit wasm binary
    let bytes = crate::emit_wasm(db, lowered)?;

    // Extract exports and imports from module
    let (exports, imports) = extract_metadata(db, module);

    Ok(WasmBinary::new(db, bytes, exports, imports))
}

/// Extract metadata (exports and imports) from a compiled module.
fn extract_metadata<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> (Vec<Symbol>, Vec<(Symbol, Symbol)>) {
    let mut exports = Vec::new();
    let mut imports = Vec::new();

    let func_dialect = Symbol::new("func");
    let func_name = Symbol::new("func");
    let wasm_dialect = Symbol::new("wasm");
    let import_func_name = Symbol::new("import_func");

    let body = module.body(db);
    for block in body.blocks(db).iter() {
        for op in block.operations(db).iter() {
            // Collect exported functions
            if op.dialect(db) == func_dialect && op.name(db) == func_name {
                if let Some(name) = get_sym_name(db, op) {
                    exports.push(name);
                }
            }
            // Collect imported functions
            else if op.dialect(db) == wasm_dialect
                && op.name(db) == import_func_name
                && let Some((module_name, func_name)) = get_import_names(db, op)
            {
                imports.push((module_name, func_name));
            }
        }
    }

    (exports, imports)
}

/// Extract sym_name attribute from an operation.
fn get_sym_name<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> Option<Symbol> {
    use trunk_ir::Attribute;

    let attrs = op.attributes(db);
    if let Some(Attribute::Symbol(sym)) = attrs.get(&ATTR_SYM_NAME()) {
        return Some(*sym);
    }
    None
}

/// Extract module and function names from an import_func operation.
fn get_import_names<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
) -> Option<(Symbol, Symbol)> {
    use trunk_ir::Attribute;

    let attrs = op.attributes(db);

    // Extract module name attribute
    let module_name = match attrs.get(&ATTR_MODULE()) {
        Some(Attribute::Symbol(s)) => Some(*s),
        _ => None,
    }?;

    // Extract function name attribute
    let func_name = match attrs.get(&ATTR_NAME()) {
        Some(Attribute::Symbol(s)) => Some(*s),
        _ => None,
    }?;

    Some((module_name, func_name))
}
