//! Language-agnostic WebAssembly module translation.
//!
//! This module provides functions for validating and emitting WebAssembly binaries
//! from TrunkIR modules that have already been lowered to the wasm dialect.

use trunk_ir::dialect::core::Module;
use trunk_ir::{Attribute, Operation, Symbol};

use crate::{CompilationResult, WasmBinary, emit_wasm, validate_wasm_ir};

// Shared attribute symbols
trunk_ir::symbols! {
    ATTR_SYM_NAME => "sym_name",
    ATTR_MODULE => "module",
    ATTR_NAME => "name",
}

/// Emit a WebAssembly binary from a lowered TrunkIR module.
///
/// This function assumes the module has already been lowered to wasm dialect
/// and all type conversions have been resolved. It:
/// 1. Validates the IR (checks for unresolved types and non-wasm ops)
/// 2. Emits the wasm binary
/// 3. Extracts metadata (exports, imports)
///
/// For Tribute-specific compilation (including lowering from high-level IR),
/// use the orchestration in the main crate's pipeline.
#[salsa::tracked]
pub fn emit_module_to_wasm<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> CompilationResult<WasmBinary<'db>> {
    // Validate IR (check for unresolved types and non-wasm ops)
    validate_wasm_ir(db, module)?;

    // Emit wasm binary
    let bytes = emit_wasm(db, module)?;

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
