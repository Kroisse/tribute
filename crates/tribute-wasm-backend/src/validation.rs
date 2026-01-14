//! IR validation for wasm backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be in the `wasm` dialect (error)
//! - All types should be resolved (warning only - emit has fallbacks)
//!
//! Dialect validation errors prevent emission from proceeding.

use tracing::warn;
use trunk_ir::dialect::core::Module;
use trunk_ir::{DialectOp, Operation, Region, Symbol, Type};

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

/// Validate that a module's IR is ready for wasm emission.
///
/// This function checks:
/// 1. All operations are in the `wasm` dialect (except allowed exceptions)
/// 2. All result types are concrete wasm types (no type variables)
///
/// Returns an error if validation fails, preventing emission.
pub fn validate_wasm_ir<'db>(
    db: &'db dyn salsa::Database,
    module: Module<'db>,
) -> CompilationResult<()> {
    let mut errors: Vec<String> = Vec::new();

    let body = module.body(db);
    validate_region(db, body, &mut errors);

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
fn validate_region<'db>(
    db: &'db dyn salsa::Database,
    region: Region<'db>,
    errors: &mut Vec<String>,
) {
    for block in region.blocks(db).iter() {
        for op in block.operations(db).iter() {
            validate_operation(db, op, errors);
        }
    }
}

/// Validate a single operation.
fn validate_operation<'db>(
    db: &'db dyn salsa::Database,
    op: &Operation<'db>,
    errors: &mut Vec<String>,
) {
    let dialect = op.dialect(db);
    let name = op.name(db);

    // Check dialect - must be wasm (with specific exceptions)
    if !is_allowed_dialect(db, op) {
        errors.push(format!("Non-wasm operation found: {}.{}", dialect, name));
    }

    // Check result types - warn about type variables but don't fail
    // TODO: Make this an error once typeck properly resolves all types (see type inference issues)
    for (idx, result_ty) in op.results(db).iter().enumerate() {
        if is_type_variable(db, *result_ty) {
            warn!(
                "Unresolved type variable in result {} of {}.{}: {} (emit will use fallback)",
                idx,
                dialect,
                name,
                format_type(db, *result_ty)
            );
        }
    }

    // Recursively validate nested regions
    for region in op.regions(db).iter() {
        validate_region(db, *region, errors);
    }
}

/// Check if an operation's dialect is allowed in the emit phase.
fn is_allowed_dialect<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> bool {
    let wasm_dialect = Symbol::new("wasm");
    let dialect = op.dialect(db);

    if dialect == wasm_dialect {
        return true;
    }

    // Allow core.module at the top level
    if trunk_ir::dialect::core::Module::matches(db, *op) {
        return true;
    }

    false
}

/// Check if a type is an unresolved type variable.
fn is_type_variable<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> bool {
    tribute_ir::dialect::tribute::is_type_var(db, ty)
}

/// Format a type for diagnostic output.
fn format_type<'db>(db: &'db dyn salsa::Database, ty: Type<'db>) -> String {
    ty.dialect(db)
        .with_str(|d| ty.name(db).with_str(|n| format!("{}.{}", d, n)))
}
