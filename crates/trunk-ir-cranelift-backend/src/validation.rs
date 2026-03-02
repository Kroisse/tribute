//! IR validation for Cranelift backend.
//!
//! This module validates that IR is ready for emission:
//! - All operations must be in the `clif` dialect (error)
//!
//! Dialect validation errors prevent emission from proceeding.

use trunk_ir::arena::context::IrContext;
use trunk_ir::arena::refs::{BlockRef, OpRef, RegionRef};
use trunk_ir::arena::rewrite::ArenaModule;
use trunk_ir::dialect::core::Module;
use trunk_ir::ir::Symbol;
use trunk_ir::{DialectOp, Operation, Region};

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
pub fn validate_clif_ir<'db>(
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

    // Check dialect - must be clif (with specific exceptions)
    if !is_allowed_dialect(db, op) {
        errors.push(format!("Non-clif operation found: {}.{}", dialect, name));
    }

    // Recursively validate nested regions
    for region in op.regions(db).iter() {
        validate_region(db, *region, errors);
    }
}

/// Check if an operation's dialect is allowed in the emit phase.
fn is_allowed_dialect<'db>(db: &'db dyn salsa::Database, op: &Operation<'db>) -> bool {
    let clif_dialect = Symbol::new("clif");
    let dialect = op.dialect(db);

    if dialect == clif_dialect {
        return true;
    }

    // Allow core.module at the top level
    if trunk_ir::dialect::core::Module::matches(db, *op) {
        return true;
    }

    false
}

// =============================================================================
// Arena IR version
// =============================================================================

/// Validate that a module's IR is ready for Cranelift emission (arena IR).
pub fn validate_clif_ir_arena(ctx: &IrContext, module: ArenaModule) -> CompilationResult<()> {
    let mut errors: Vec<String> = Vec::new();

    if let Some(body) = module.body(ctx) {
        validate_region_arena(ctx, body, &mut errors);
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

fn validate_region_arena(ctx: &IrContext, region: RegionRef, errors: &mut Vec<String>) {
    let region_data = ctx.region(region);
    for &block in &region_data.blocks {
        validate_block_arena(ctx, block, errors);
    }
}

fn validate_block_arena(ctx: &IrContext, block: BlockRef, errors: &mut Vec<String>) {
    let block_data = ctx.block(block);
    for &op in &block_data.ops {
        validate_operation_arena(ctx, op, errors);
    }
}

fn validate_operation_arena(ctx: &IrContext, op: OpRef, errors: &mut Vec<String>) {
    let op_data = ctx.op(op);
    let dialect = op_data.dialect;
    let name = op_data.name;

    if !is_allowed_dialect_arena(ctx, op) {
        errors.push(format!("Non-clif operation found: {}.{}", dialect, name));
    }

    // Recursively validate nested regions
    for &region in &op_data.regions {
        validate_region_arena(ctx, region, errors);
    }
}

fn is_allowed_dialect_arena(ctx: &IrContext, op: OpRef) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use salsa_test_macros::salsa_test;
    use trunk_ir::dialect::{clif, core};
    use trunk_ir::{Block, BlockId, DialectOp, DialectType, Location, PathId, Span, Symbol, idvec};

    fn test_location(db: &dyn salsa::Database) -> Location<'_> {
        let path = PathId::new(db, "file:///test.trb".to_owned());
        Location::new(path, Span::new(0, 0))
    }

    #[salsa::tracked]
    fn make_valid_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        let iconst_op = clif::iconst(db, location, i64_ty, 42).as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![iconst_op]);
        let region = trunk_ir::Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_validate_valid_module(db: &salsa::DatabaseImpl) {
        let module = make_valid_module(db);
        assert!(validate_clif_ir(db, module).is_ok());
    }

    #[salsa::tracked]
    fn make_invalid_module(db: &dyn salsa::Database) -> Module<'_> {
        let location = test_location(db);
        let i64_ty = core::I64::new(db).as_type();

        // Use a func.constant (non-clif) operation
        let const_op =
            trunk_ir::dialect::func::constant(db, location, i64_ty, Symbol::new("some_func"))
                .as_operation();

        let block = Block::new(db, BlockId::fresh(), location, idvec![], idvec![const_op]);
        let region = trunk_ir::Region::new(db, location, idvec![block]);
        Module::create(db, location, "test".into(), region)
    }

    #[salsa_test]
    fn test_validate_rejects_non_clif_ops(db: &salsa::DatabaseImpl) {
        let module = make_invalid_module(db);
        let result = validate_clif_ir(db, module);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Non-clif operation found"));
    }
}
