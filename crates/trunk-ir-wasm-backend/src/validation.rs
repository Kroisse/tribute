//! IR validation for wasm backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be in the `wasm` dialect (error)
//!
//! Dialect validation errors prevent emission from proceeding.

use trunk_ir::Symbol;
use trunk_ir::arena::IrContext;
use trunk_ir::arena::Module;
use trunk_ir::arena::refs::{OpRef, RegionRef};

use crate::{CompilationError, CompilationResult};

/// Validation error details.
#[derive(Debug)]
pub struct ValidationError {
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Validate that a module's IR is ready for wasm emission (arena version).
///
/// This function checks that all operations are in the `wasm` dialect
/// (except allowed exceptions like `core.module`).
///
/// Returns an error if validation fails, preventing emission.
pub fn validate_wasm_ir(ctx: &IrContext, module: Module) -> CompilationResult<()> {
    let mut errors: Vec<String> = Vec::new();

    let body = module
        .body(ctx)
        .ok_or_else(|| CompilationError::invalid_module("module has no body region"))?;
    validate_region(ctx, body, 0, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        let message = format!(
            "IR validation failed with {} error(s):\n  - {}",
            errors.len(),
            errors.join("\n  - ")
        );
        Err(CompilationError::ir_validation(message))
    }
}

/// Validate a region recursively.
fn validate_region(ctx: &IrContext, region: RegionRef, depth: usize, errors: &mut Vec<String>) {
    for &block_ref in &ctx.region(region).blocks {
        for &op in &ctx.block(block_ref).ops {
            validate_operation(ctx, op, depth, errors);
        }
    }
}

/// Validate a single operation.
fn validate_operation(ctx: &IrContext, op: OpRef, depth: usize, errors: &mut Vec<String>) {
    let op_data = ctx.op(op);
    let dialect = op_data.dialect;
    let name = op_data.name;

    // Check dialect - must be wasm (with specific exceptions)
    if !is_allowed_dialect(ctx, op, depth) {
        errors.push(format!("Non-wasm operation found: {}.{}", dialect, name));
    }

    // Recursively validate nested regions
    for &region in &op_data.regions {
        validate_region(ctx, region, depth + 1, errors);
    }
}

/// Check if an operation's dialect is allowed in the emit phase.
fn is_allowed_dialect(ctx: &IrContext, op: OpRef, depth: usize) -> bool {
    let wasm_dialect = Symbol::new("wasm");
    let op_data = ctx.op(op);

    if op_data.dialect == wasm_dialect {
        return true;
    }

    // Allow core.module only at the top level (depth 0)
    if depth == 0 && op_data.dialect == Symbol::new("core") && op_data.name == Symbol::new("module")
    {
        return true;
    }

    false
}
