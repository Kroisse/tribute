//! IR validation for Cranelift backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be in the `clif` dialect (error)
//!
//! Dialect validation errors prevent emission from proceeding.

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::{BlockRef, OpRef, RegionRef};
use trunk_ir::rewrite::Module;

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

/// Validate that a module's IR is ready for Cranelift emission.
///
/// This function checks that all operations are in the `clif` dialect
/// (except allowed exceptions like `core.module`).
///
/// Returns an error if validation fails, preventing emission.
pub fn validate_clif_ir(ctx: &IrContext, module: Module) -> CompilationResult<()> {
    let mut errors: Vec<String> = Vec::new();

    if let Some(body) = module.body(ctx) {
        validate_region(ctx, body, &mut errors);
    } else {
        errors.push("Module has no body region".to_string());
    }

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

fn validate_region(ctx: &IrContext, region: RegionRef, errors: &mut Vec<String>) {
    let region_data = ctx.region(region);
    for &block in &region_data.blocks {
        validate_block(ctx, block, errors);
    }
}

fn validate_block(ctx: &IrContext, block: BlockRef, errors: &mut Vec<String>) {
    let block_data = ctx.block(block);
    for &op in &block_data.ops {
        validate_operation(ctx, op, errors);
    }
}

fn validate_operation(ctx: &IrContext, op: OpRef, errors: &mut Vec<String>) {
    let op_data = ctx.op(op);
    let dialect = op_data.dialect;
    let name = op_data.name;

    if !is_allowed_dialect(ctx, op) {
        errors.push(format!("Non-clif operation found: {}.{}", dialect, name));
    }

    // Recursively validate nested regions
    for &region in &op_data.regions {
        validate_region(ctx, region, errors);
    }
}

fn is_allowed_dialect(ctx: &IrContext, op: OpRef) -> bool {
    let clif_dialect = Symbol::new("clif");
    let core_dialect = Symbol::new("core");
    let module_name = Symbol::new("module");
    let op_data = ctx.op(op);

    if op_data.dialect == clif_dialect {
        return true;
    }

    // Allow core.module at the top level
    if op_data.dialect == core_dialect && op_data.name == module_name {
        return true;
    }

    false
}
