//! Compilation from TrunkIR to WebAssembly binary.
//!
//! This module provides the main entry point for compiling a Tribute TrunkIR module
//! to a WebAssembly binary. It combines lowering (IR transformation) and emission
//! (binary generation) into a single tracked artifact.

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
/// 2. Emits the wasm dialect to a WebAssembly binary
/// 3. Collects metadata (exports, imports) for tooling integration
///
/// Memoization is automatic via Salsa's incremental compilation system.
#[salsa::tracked]
pub fn compile_to_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> crate::CompilationResult<WasmBinary<'db>> {
    // TODO: Phase 1 - Lower to wasm dialect
    // CRITICAL: The module currently contains mid-level IR (func, arith, scf, etc.)
    // but emit_wasm expects wasm dialect operations.
    // Until the lowering pass is implemented in tribute-passes, this will fail
    // for any module containing non-func/wasm operations.
    // Future: Call lower_to_wasm(db, module) here before emit_wasm

    let bytes = crate::emit_wasm(db, module)?;

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
    for (_, attr) in attrs.iter() {
        if let Attribute::Symbol(sym) = attr {
            return Some(*sym);
        }
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
    let mut module_name = None;
    let mut func_name = None;

    for (_, attr) in attrs.iter() {
        if let Attribute::Symbol(sym) = attr {
            // First symbol is typically module, second is function
            if module_name.is_none() {
                module_name = Some(*sym);
            } else if func_name.is_none() {
                func_name = Some(*sym);
                break;
            }
        }
    }

    match (module_name, func_name) {
        (Some(m), Some(f)) => Some((m, f)),
        _ => None,
    }
}
