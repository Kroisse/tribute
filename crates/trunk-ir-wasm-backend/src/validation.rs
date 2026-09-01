//! IR validation for wasm backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be in the `wasm` dialect (error)
//!
//! Dialect validation errors prevent emission from proceeding.

use trunk_ir::IrContext;
use trunk_ir::Module;
use trunk_ir::Symbol;
use trunk_ir::dialect::wasm as wasm_dialect;
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{OpRef, RegionRef};

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
    validate_return_call_indirect(ctx, op, errors);

    // Recursively validate nested regions
    for &region in &op_data.regions {
        validate_region(ctx, region, depth + 1, errors);
    }
}

/// Validate the source-of-truth signature needed for a proper indirect tail
/// transfer before emission has a chance to construct a type section.
fn validate_return_call_indirect(ctx: &IrContext, op: OpRef, errors: &mut Vec<String>) {
    if !wasm_dialect::ReturnCallIndirect::matches(ctx, op) {
        return;
    }
    if let Err(error) = crate::emit::helpers::exact_return_call_indirect_signature(ctx, op) {
        errors.push(error.to_string());
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

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;

    #[test]
    fn rejects_invalid_return_call_indirect_exact_contracts() {
        let rejects = |source: &str, diagnostic: &str| {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, source);
            let error = validate_wasm_ir(&ctx, module).unwrap_err();
            assert!(error.to_string().contains(diagnostic), "{error}");
        };

        rejects(
            r#"core.module @test {
  wasm.func @caller(%table_index: core.i32, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table_index, %value {table = 0, type_idx = 0}
  }
}"#,
            "lacks signature",
        );

        rejects(
            r#"core.module @test {
  wasm.func @caller(%table_index: core.i32, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = core.i32, table = 0, type_idx = 0}
  }
}"#,
            "signature must be core.func",
        );

        rejects(
            r#"core.module @test {
  wasm.func @caller(%table_index: core.i32, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = core.func(core.i32, core.i32), table = 0, type_idx = 0}
  }
}"#,
            "must have an empty result",
        );

        rejects(
            r#"core.module @test {
  wasm.func @caller() -> core.nil {
    wasm.return_call_indirect {signature = core.func(core.nil), table = 0, type_idx = 0}
  }
}"#,
            "requires a table index operand",
        );

        rejects(
            r#"core.module @test {
  wasm.func @caller(%table_index: core.i64, %value: core.i32) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = core.func(core.nil, core.i32), table = 0, type_idx = 0}
  }
}"#,
            "first operand must be an i32 table index",
        );

        rejects(
            r#"core.module @test {
  wasm.func @caller(%table_index: core.i32, %value: core.i64) -> core.nil {
    wasm.return_call_indirect %table_index, %value {signature = core.func(core.nil, core.i32), table = 0, type_idx = 0}
  }
}"#,
            "operands do not match its exact signature",
        );
    }
}
